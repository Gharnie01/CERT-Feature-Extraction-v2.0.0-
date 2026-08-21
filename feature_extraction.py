#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CERT Insider Threat Test Dataset -- Feature Extraction (modernised)
================================================================================

Supported releases : r4.1, r4.2, r5.1, r5.2, r6.1, r6.2
Supported modes    : week, day, session, subsession, all

--------------------------------------------------------------------------------
RELATIONSHIP TO THE ORIGINAL
--------------------------------------------------------------------------------
This is a performance/reliability modernisation of `feature_extraction.py` from

    https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets

Original research and feature-extraction design:
    D. C. Le, N. Zincir-Heywood and M. I. Heywood,
    "Analyzing Data Granularity Levels for Insider Threat Detection Using
     Machine Learning", IEEE TNSM, 17(1):30-44, March 2020.
    doi:10.1109/TNSM.2020.2967721

Dataset:
    Lindauer, Brian (2020): Insider Threat Test Dataset. Carnegie Mellon
    University. https://doi.org/10.1184/R1/12841247.v1
    J. Glasser and B. Lindauer, "Bridging the Gap...", IEEE SPW 2013.

Modernisation and engineering work:
    Gharnie01  --  https://github.com/Gharnie01

Original licence: MIT. Please cite the TNSM 2020 paper if you use this script
or the data it produces. See README.md for citation details.

--------------------------------------------------------------------------------
WHAT CHANGED, AND WHAT DID NOT
--------------------------------------------------------------------------------
NOT changed -- every function in PART A below is the original feature logic,
kept line-for-line wherever possible. Feature names, feature order, column
order, label semantics and numeric values are intended to be IDENTICAL to the
original. The only edits in PART A are compatibility fixes for pandas >= 2.0
(the original cannot run at all on a modern stack), each marked `# COMPAT:`.

Changed -- PART B, the orchestration around that logic:

  1. Step 1 date parsing.  The original called datetime.strptime() once per log
     line to decide which week a line belongs to. On r6.2 that is ~135 million
     strptime calls. Replaced with a lookup table keyed on the 'MM/DD/YYYY'
     prefix: ~516 distinct dates, so ~516 parses instead of 135 million.
     Exactly equivalent, since the week number depends only on the date part.

  2. Step 1 accumulation.  `DataFrame.append` in a loop (quadratic, and removed
     in pandas 2.0) replaced with list accumulation + a single pd.concat.

  3. Step 3 row access.  The original indexed `df_acts_u.iloc[i]` inside a
     per-action loop; each call builds a fresh Series. Rows are now converted
     to plain dicts once per user. The feature functions receive a mapping and
     behave identically.

  4. Step 3 user selection.  `acts_week[acts_week.user == u]` inside a loop over
     users is a full scan per user. Replaced with a single groupby.

  5. Step 3 USB connect/disconnect lookup.  The original re-filtered the whole
     remaining tail of the user's week for every Connect action -- quadratic in
     actions per user. Replaced with per-PC position arrays and binary search.
     Semantics preserved exactly, including the quirk that the search window
     starts AT the current action (see the note in _connect_duration).

  6. Step 4 grouping.  `w[w['user']==v]` and `uactw[uactw['day']==d]` inside
     loops replaced with groupby. Per-user metadata rows are precomputed once
     instead of being re-sliced per instance.

  7. Reliability.  SQLite checkpointing per (mode, week) partition, --resume,
     atomic write-then-rename, output validation before a partition is marked
     complete, isolated partition failure, intermediates retained by default.

  8. Selective work.  --mode day computes ONLY day. The original always ran
     week + day + session + subsession.

  9. Observability.  Rich progress if available (auto-detected, --no-rich to
     disable), standard logging, per-partition timing, run_metadata.json.

 10. Portability.  wget/rm/head shelled out to the OS; replaced with urllib,
     shutil and pure-Python I/O. Runs on Linux, macOS and Windows/WSL.

Parallel backend remains joblib/loky, as in the original. It is not replaced
speculatively; --benchmark exists to measure worker counts on real partitions.

--------------------------------------------------------------------------------
QUICK START (day-granularity workflow)
--------------------------------------------------------------------------------
    cd /path/to/r4.2
    python feature_extraction.py --mode day --workers 8

    # interrupted? just re-run; completed weeks are skipped
    python feature_extraction.py --mode day --workers 8 --resume

Output: ExtractedData/dayr4.2.csv
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import platform
import re
import shutil
import sqlite3
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

__version__ = "2.0.0"

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

VALID_DATASETS = ["r4.1", "r4.2", "r5.1", "r5.2", "r6.1", "r6.2"]
R4 = ["r4.1", "r4.2"]                      # 27 numeric cols, no project field
R5 = ["r5.1", "r5.2"]
R6 = ["r6.1", "r6.2"]

# Only 73 weeks of data exist in the r4.x releases; 75 in r5.x/r6.x.
NUM_WEEKS = {"r4.1": 73, "r4.2": 73, "r5.1": 75, "r5.2": 75, "r6.1": 75, "r6.2": 75}

# Original defaults. Do not change without also changing the output filenames,
# which encode these values (e.g. sessionnact25r5.2.csv).
DEFAULT_SUBSESSION_TIME = [120, 240]       # minutes
DEFAULT_SUBSESSION_NACT = [25, 50]         # actions

ALL_MODES = ["week", "day", "session", "subsession"]

# Informational (non-feature) columns, per the original README. `insider` is the
# TARGET, never an informational field, and is never dropped.
INFORMATIONAL_FIELDS = ["subs_ind", "starttime", "endtime", "sessionid",
                        "user", "day", "week"]

ANSWERS_URL = "https://kilthub.cmu.edu/ndownloader/files/24857828"

log = logging.getLogger("cert-extract")


# ──────────────────────────────────────────────────────────────────────────────
# Progress reporting -- rich when available, plain logging otherwise
# ──────────────────────────────────────────────────────────────────────────────

try:
    from rich.progress import (Progress, BarColumn, TextColumn, TimeElapsedColumn,
                               TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn)
    from rich.console import Console
    _RICH_AVAILABLE = True
except ImportError:                                     # pragma: no cover
    _RICH_AVAILABLE = False


class ProgressReporter:
    """
    Thin wrapper so the pipeline code never has to care whether rich is present.

    Rich gives a live bar with throughput and an ETA computed from observed
    partition durations. Without rich (or with --no-rich, or when stdout is not
    a TTY, e.g. piped to a log file) it degrades to periodic log lines carrying
    the same information. Nothing in the pipeline changes either way.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and _RICH_AVAILABLE and sys.stdout.isatty()
        self._progress = None
        self._tasks = {}

    def __enter__(self):
        if self.enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TextColumn("{task.fields[rate]}"),
                TimeElapsedColumn(),
                TextColumn("eta"),
                TimeRemainingColumn(),
                console=Console(stderr=False),
                transient=False,
            )
            self._progress.start()
        return self

    def __exit__(self, *exc):
        if self._progress is not None:
            self._progress.stop()
        return False

    def add(self, key: str, description: str, total: int):
        if self._progress is not None:
            self._tasks[key] = self._progress.add_task(description, total=total, rate="")
        else:
            log.info("%s: 0/%d", description, total)
            self._tasks[key] = {"desc": description, "done": 0,
                                "total": total, "t0": time.time()}

    def advance(self, key: str, n: int = 1, rate_hint: str = ""):
        if self._progress is not None:
            self._progress.update(self._tasks[key], advance=n, rate=rate_hint)
        else:
            t = self._tasks[key]
            t["done"] += n
            # Plain mode: log every 10 partitions so a piped log stays readable
            # but still proves the process is alive.
            if t["done"] % 10 == 0 or t["done"] == t["total"]:
                elapsed = time.time() - t["t0"]
                rate = t["done"] / elapsed * 60 if elapsed > 0 else 0
                remaining = (t["total"] - t["done"]) / rate * 60 if rate > 0 else 0
                log.info("%s: %d/%d  (%.1f/min, eta %.0fs)",
                         t["desc"], t["done"], t["total"], rate, remaining)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint store
# ──────────────────────────────────────────────────────────────────────────────

class CheckpointDB:
    """
    SQLite record of every independently processable unit.

    The unit is (stage, mode, partition) where partition is the week number.
    Weeks are independent in every stage, so this is the smallest granularity
    that does not require restructuring the original algorithm.

    A partition is COMPLETED only after its output has been written, flushed,
    validated and atomically renamed -- never merely because a filename exists.
    A crash mid-write therefore leaves a .tmp file and a 'running' row, and the
    work is correctly redone on resume.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS partitions (
        dataset          TEXT NOT NULL,
        stage            TEXT NOT NULL,
        mode             TEXT NOT NULL,
        partition        INTEGER NOT NULL,
        status           TEXT NOT NULL,
        started_at       TEXT,
        completed_at     TEXT,
        duration_seconds REAL,
        rows             INTEGER,
        output_path      TEXT,
        error            TEXT,
        PRIMARY KEY (dataset, stage, mode, partition)
    );
    """

    def __init__(self, path: Path, dataset: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self._conn = sqlite3.connect(str(self.path), timeout=60.0)

        # WAL journalling needs a shared-memory (-shm) file backed by mmap.
        # DrvFs/9p -- i.e. anything under /mnt/c on WSL -- does not support
        # that, and SQLite can block indefinitely rather than fail cleanly.
        # Fall back to rollback journalling there; it is slower per commit but
        # the commit rate here is one row per partition, so it costs nothing.
        if self._on_windows_mount():
            self._conn.execute("PRAGMA journal_mode=DELETE")
            log.debug("checkpoint DB on a Windows mount -- WAL disabled")
        else:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def _on_windows_mount(self) -> bool:
        """True if the DB lives on a DrvFs/9p mount (WSL's /mnt/<drive>)."""
        try:
            p = str(self.path.resolve())
            if not p.startswith("/mnt/"):
                return False
            # /mnt/c/... is a Windows drive; /mnt/data on a native disk is not.
            with open("/proc/mounts") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) > 2 and p.startswith(parts[1]) \
                            and parts[2] in ("9p", "drvfs", "drvfs2", "cifs"):
                        return True
            # /proc/mounts unavailable or inconclusive: assume the worst for
            # anything under /mnt/<single letter>/.
            seg = p.split("/")
            return len(seg) > 2 and len(seg[2]) == 1
        except Exception:
            return False

    def completed(self, stage: str, mode: str) -> set:
        cur = self._conn.execute(
            "SELECT partition FROM partitions "
            "WHERE dataset=? AND stage=? AND mode=? AND status='completed'",
            (self.dataset, stage, mode))
        return {row[0] for row in cur.fetchall()}

    def mark(self, stage: str, mode: str, partition: int, status: str,
             duration: float = None, rows: int = None,
             output_path: str = None, error: str = None):
        now = datetime.now().isoformat(timespec="seconds")
        self._conn.execute(
            "INSERT INTO partitions "
            "(dataset,stage,mode,partition,status,started_at,completed_at,"
            " duration_seconds,rows,output_path,error) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(dataset,stage,mode,partition) DO UPDATE SET "
            "  status=excluded.status, completed_at=excluded.completed_at,"
            "  duration_seconds=excluded.duration_seconds, rows=excluded.rows,"
            "  output_path=excluded.output_path, error=excluded.error",
            (self.dataset, stage, mode, partition, status, now,
             now if status in ("completed", "failed") else None,
             duration, rows, output_path, error))
        self._conn.commit()

    def summary(self, stage: str, mode: str) -> dict:
        cur = self._conn.execute(
            "SELECT status, COUNT(*) FROM partitions "
            "WHERE dataset=? AND stage=? AND mode=? GROUP BY status",
            (self.dataset, stage, mode))
        return dict(cur.fetchall())

    def durations(self, stage: str, mode: str) -> list:
        cur = self._conn.execute(
            "SELECT partition, duration_seconds FROM partitions "
            "WHERE dataset=? AND stage=? AND mode=? AND status='completed' "
            "ORDER BY duration_seconds DESC",
            (self.dataset, stage, mode))
        return cur.fetchall()

    def close(self):
        self._conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Atomic output helpers
# ──────────────────────────────────────────────────────────────────────────────

def atomic_write_pickle(df: pd.DataFrame, target: Path) -> int:
    """
    Write, flush, validate, then rename. The rename is atomic on POSIX and on
    Windows via os.replace, so a reader can never observe a half-written file
    and a checkpoint can never point at one.

    Returns the row count, which the caller records in the checkpoint DB.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")

    df.to_pickle(tmp)

    # Validation: re-read and confirm shape. Cheap relative to producing it,
    # and it is the difference between "the file exists" and "the file is good".
    check = pd.read_pickle(tmp)
    if check.shape != df.shape:
        tmp.unlink(missing_ok=True)
        raise IOError(f"validation failed for {target}: "
                      f"wrote {df.shape}, read back {check.shape}")

    os.replace(tmp, target)
    return len(df)


def ensure_answers(work_dir: Path):
    """
    Fetch and unpack the ground-truth 'answers' archive if absent.

    The original shelled out to wget and tar, which restricts it to Linux with
    those binaries installed. urllib + tarfile is equivalent and portable.
    """
    if (work_dir / "answers").is_dir():
        return
    archive = work_dir / "answers.tar.bz2"
    log.info("answers/ not found -- downloading ground truth")
    if not archive.is_file():
        with urllib.request.urlopen(ANSWERS_URL) as resp, open(archive, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(path=work_dir)
    log.info("answers/ ready")


# ══════════════════════════════════════════════════════════════════════════════
#
#  PART A -- ORIGINAL FEATURE LOGIC
#
#  Everything below to the PART B banner defines WHAT is computed. It is the
#  original code. Do not "improve" it: any change here changes the extracted
#  features and breaks comparability with the published results and with every
#  paper that used this extractor.
#
#  Edits are limited to pandas>=2.0 compatibility and are marked `# COMPAT:`.
#
# ══════════════════════════════════════════════════════════════════════════════

def time_convert(inp, mode, real_sd='2010-01-02', sd_monday="2009-12-28"):
    if mode == 'e2t':
        return datetime.fromtimestamp(inp).strftime('%m/%d/%Y %H:%M:%S')
    elif mode == 't2e':
        # COMPAT: strftime('%s') is glibc-only and silently wrong elsewhere.
        return str(int(datetime.strptime(inp, '%m/%d/%Y %H:%M:%S').timestamp()))
    elif mode == 't2dt':
        return datetime.strptime(inp, '%m/%d/%Y %H:%M:%S')
    elif mode == 't2date':
        return datetime.strptime(inp, '%m/%d/%Y %H:%M:%S').strftime("%Y-%m-%d")
    elif mode == 'dt2t':
        return inp.strftime('%m/%d/%Y %H:%M:%S')
    elif mode == 'dt2W':
        return int(inp.strftime('%W'))
    elif mode == 'dt2d':
        return inp.strftime('%m/%d/%Y %H:%M:%S')
    elif mode == 'dt2date':
        return inp.strftime("%Y-%m-%d")
    elif mode == 'dt2dn':                       # datetime to day number
        startdate = datetime.strptime(sd_monday, '%Y-%m-%d')
        return (inp - startdate).days
    elif mode == 'dn2epoch':                    # datenum to epoch
        dt = datetime.strptime(sd_monday, '%Y-%m-%d') + timedelta(days=inp)
        return int(dt.timestamp())
    elif mode == 'dt2wn':                       # datetime to week number
        startdate = datetime.strptime(real_sd, '%Y-%m-%d')
        return (inp - startdate).days // 7
    elif mode == 't2wn':                        # timestamp string to week number
        startdate = datetime.strptime(real_sd, '%Y-%m-%d')
        return (datetime.strptime(inp, '%m/%d/%Y %H:%M:%S') - startdate).days // 7
    elif mode == 'dt2wd':
        return int(inp.strftime("%w"))
    elif mode == 'm2dt':
        return datetime.strptime(inp, "%Y-%m")
    elif mode == 'datetoweekday':
        return int(datetime.strptime(inp, "%Y-%m-%d").strftime('%w'))
    elif mode == 'datetoweeknum':
        w0 = datetime.strptime(sd_monday, "%Y-%m-%d")
        return int((datetime.strptime(inp, "%Y-%m-%d") - w0).days / 7)
    elif mode == 'weeknumtodate':
        startday = datetime.strptime(sd_monday, "%Y-%m-%d")
        return startday + timedelta(weeks=inp)


def process_user_pc(upd, roles):
    """Figure out which PC belongs to which user (original logic)."""
    upd['sharedpc'] = None
    upd['npc'] = upd['pcs'].apply(lambda x: len(x))
    # COMPAT: .at does not accept a boolean mask; the original relied on an
    # accident of older pandas. .loc is the correct accessor and is equivalent.
    upd.loc[upd['npc'] == 1, 'pc'] = upd[upd['npc'] == 1]['pcs'].apply(lambda x: x[0])
    multiuser_pcs = np.concatenate(upd[upd['npc'] > 1]['pcs'].values).tolist()
    set_multiuser_pc = list(set(multiuser_pcs))
    count = {}
    for pc in set_multiuser_pc:
        count[pc] = multiuser_pcs.count(pc)
    for u in upd[upd['npc'] > 1].index:
        sharedpc = upd.loc[u]['pcs']
        count_u_pc = [count[pc] for pc in upd.loc[u]['pcs']]
        the_pc = count_u_pc.index(min(count_u_pc))
        upd.at[u, 'pc'] = sharedpc[the_pc]
        if roles.loc[u] != 'ITAdmin':
            sharedpc.remove(sharedpc[the_pc])
        upd.at[u, 'sharedpc'] = sharedpc
    return upd


def getuserlist(dname='r4.2', psycho=True, week_dir="DataByWeek"):
    allfiles = ['LDAP/' + f1 for f1 in os.listdir('LDAP')
                if os.path.isfile('LDAP/' + f1)]
    alluser = {}
    alreadyFired = []
    for file in allfiles:
        af = (pd.read_csv(file, delimiter=',')).values
        employeesThisMonth = []
        for i in range(len(af)):
            employeesThisMonth.append(af[i][1])
            if af[i][1] not in alluser:
                alluser[af[i][1]] = (af[i][0:1].tolist() + af[i][2:].tolist()
                                     + [file.split('.')[0], np.nan])
        firedEmployees = list(set(alluser.keys()) - set(alreadyFired)
                              - set(employeesThisMonth))
        alreadyFired = alreadyFired + firedEmployees
        for e in firedEmployees:
            alluser[e][-1] = file.split('.')[0]

    if psycho and os.path.isfile("psychometric.csv"):
        p_score = pd.read_csv("psychometric.csv", delimiter=',').values
        for id in range(len(p_score)):
            alluser[p_score[id, 1]] = alluser[p_score[id, 1]] + list(p_score[id, 2:])
        df = pd.DataFrame.from_dict(alluser, orient='index')
        if dname in R4:
            df.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept',
                          'team', 'sup', 'wstart', 'wend', 'O', 'C', 'E', 'A', 'N']
        elif dname in R5 + R6:
            df.columns = ['uname', 'email', 'role', 'project', 'b_unit', 'f_unit',
                          'dept', 'team', 'sup', 'wstart', 'wend',
                          'O', 'C', 'E', 'A', 'N']
    else:
        df = pd.DataFrame.from_dict(alluser, orient='index')
        if dname in R4:
            df.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept',
                          'team', 'sup', 'wstart', 'wend']
        elif dname in R5 + R6:
            df.columns = ['uname', 'email', 'role', 'project', 'b_unit', 'f_unit',
                          'dept', 'team', 'sup', 'wstart', 'wend']

    df['pc'] = None
    for i in df.index:
        if type(df.loc[i]['sup']) == str:
            sup = df[df['uname'] == df.loc[i]['sup']].index[0]
        else:
            sup = None
        df.at[i, 'sup'] = sup

    # Read the first 2 weeks to determine each user's PC.
    w1 = pd.read_pickle(f"{week_dir}/1.pickle")
    w2 = pd.read_pickle(f"{week_dir}/2.pickle")
    user_pc_dict = pd.DataFrame(index=df.index)
    user_pc_dict['pcs'] = None
    for u in df.index:
        pc = list(set(w1[w1['user'] == u]['pc']) & set(w2[w2['user'] == u]['pc']))
        user_pc_dict.at[u, 'pcs'] = pc

    upd = process_user_pc(user_pc_dict, df['role'])
    df['pc'] = upd['pc']
    df['sharedpc'] = upd['sharedpc']
    return df


def get_mal_userdata(data='r4.2', usersdf=None, work_dir=None, week_dir="DataByWeek"):
    # COMPAT: original used wget + tar via os.system (Linux-only).
    ensure_answers(Path(work_dir) if work_dir else Path.cwd())

    listmaluser = pd.read_csv("answers/insiders.csv")
    listmaluser['dataset'] = listmaluser['dataset'].apply(lambda x: str(x))
    listmaluser = listmaluser[listmaluser['dataset'] == data.replace("r", "")]

    # For r6.2, the new time in the scenario 4 answer file is incomplete.
    if data == 'r6.2':
        listmaluser.loc[listmaluser['scenario'] == 4, 'start'] = \
            '02' + listmaluser[listmaluser['scenario'] == 4]['start']

    # COMPAT: DataFrame.applymap is deprecated in favour of DataFrame.map.
    _parse = lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S")
    for _c in ['start', 'end']:
        listmaluser[_c] = listmaluser[_c].map(_parse)

    if type(usersdf) != pd.core.frame.DataFrame:
        usersdf = getuserlist(data, week_dir=week_dir)

    usersdf['malscene'] = 0
    usersdf['mstart'] = None
    usersdf['mend'] = None
    usersdf['malacts'] = None
    for i in listmaluser.index:
        usersdf.loc[listmaluser['user'][i], 'mstart'] = listmaluser['start'][i]
        usersdf.loc[listmaluser['user'][i], 'mend'] = listmaluser['end'][i]
        usersdf.loc[listmaluser['user'][i], 'malscene'] = listmaluser['scenario'][i]
        if data in ['r4.2', 'r5.2']:
            malacts = open(f"answers/r{listmaluser['dataset'][i]}-"
                           f"{listmaluser['scenario'][i]}/"
                           + listmaluser['details'][i], 'r').read().strip().split("\n")
        else:                       # only 1 malicious user per scenario, no folder
            malacts = open("answers/" + listmaluser['details'][i],
                           'r').read().strip().split("\n")
        malacts = [x.split(',') for x in malacts]
        mal_users = np.array([x[3].strip('"') for x in malacts])
        mal_act_ids = np.array([x[1].strip('"') for x in malacts])
        usersdf.at[listmaluser['user'][i], 'malacts'] = \
            mal_act_ids[mal_users == listmaluser['user'][i]]
    return usersdf


def is_after_whour(dt):
    """Workhours assumed 07:30-17:30."""
    wday_start = datetime.strptime("7:30", "%H:%M").time()
    wday_end = datetime.strptime("17:30", "%H:%M").time()
    dt = dt.time()
    if dt < wday_start or dt > wday_end:
        return True
    return False


def is_weekend(dt):
    if dt.strftime("%w") in ['0', '6']:
        return True
    return False


def email_process(act, data='r4.2', separate_send_receive=True):
    receivers = act['to'].split(';')
    if type(act['cc']) == str:
        receivers = receivers + act['cc'].split(";")
    if type(act['bcc']) == str:
        bccreceivers = act['bcc'].split(";")
    else:
        bccreceivers = []
    exemail = False
    n_exdes = 0
    for i in receivers + bccreceivers:
        if 'dtaa.com' not in i:
            exemail = True
            n_exdes += 1
    n_des = len(receivers) + len(bccreceivers)
    Xemail = 1 if exemail else 0
    n_bccdes = len(bccreceivers)
    exbccmail = 0
    email_text_len = len(act['content'])
    email_text_nwords = act['content'].count(' ') + 1
    for i in bccreceivers:
        if 'dtaa.com' not in i:
            exbccmail = 1
            break

    if data in R5 + R6:
        send_mail = 1 if act['activity'] == 'Send' else 0
        receive_mail = 1 if act['activity'] in ['Receive', 'View'] else 0
        atts = act['att'].split(';')
        n_atts = len(atts)
        size_atts = 0
        att_types = [0, 0, 0, 0, 0, 0]
        att_sizes = [0, 0, 0, 0, 0, 0]
        for att in atts:
            if '.' in att:
                tmp = file_process(att, filetype='att')
                att_types = [sum(x) for x in zip(att_types, tmp[0])]
                att_sizes = [sum(x) for x in zip(att_sizes, tmp[1])]
                size_atts += sum(tmp[1])
        return ([send_mail, receive_mail, n_des, n_atts, Xemail, n_exdes,
                 n_bccdes, exbccmail, int(act['size']), email_text_len,
                 email_text_nwords] + att_types + att_sizes)
    elif data in R4:
        return [n_des, int(act['#att']), Xemail, n_exdes, n_bccdes, exbccmail,
                int(act['size']), email_text_len, email_text_nwords]


def http_process(act, data='r4.2'):
    url_len = len(act['url/fname'])
    url_depth = act['url/fname'].count('/') - 2
    content_len = len(act['content'])
    content_nwords = act['content'].count(' ') + 1

    domainname = re.findall("//(.*?)/", act['url/fname'])[0]
    domainname.replace("www.", "")
    dn = domainname.split(".")
    if len(dn) > 2 and not any([x in domainname for x in
                                ["google.com", '.co.uk', '.co.nz', 'live.com']]):
        domainname = ".".join(dn[-2:])

    # categories: other 1, socnet 2, cloud 3, job 4, leak 5, hack 6
    if domainname in ['dropbox.com', 'drive.google.com', 'mega.co.nz',
                      'account.live.com']:
        r = 3
    elif domainname in ['wikileaks.org', 'freedom.press', 'theintercept.com']:
        r = 5
    elif domainname in ['facebook.com', 'twitter.com', 'plus.google.com',
                        'instagr.am', 'instagram.com', 'flickr.com',
                        'linkedin.com', 'reddit.com', 'about.com', 'youtube.com',
                        'pinterest.com', 'tumblr.com', 'quora.com', 'vine.co',
                        'match.com', 't.co']:
        r = 2
    elif domainname in ['indeed.com', 'monster.com', 'careerbuilder.com',
                        'simplyhired.com']:
        r = 4
    elif (('job' in domainname and ('hunt' in domainname or 'search' in domainname))
          or ('aol.com' in domainname and ("recruit" in act['url/fname']
                                           or "job" in act['url/fname']))):
        r = 4
    elif domainname in ['webwatchernow.com', 'actionalert.com', 'relytec.com',
                        'refog.com', 'wellresearchedreviews.com',
                        'softactivity.com', 'spectorsoft.com',
                        'best-spy-soft.com']:
        r = 6
    elif 'keylog' in domainname:
        r = 6
    else:
        r = 1

    if data in R6:
        http_act_dict = {'www visit': 1, 'www download': 2, 'www upload': 3}
        http_act = http_act_dict.get(act['activity'].lower(), 0)
        return [r, url_len, url_depth, content_len, content_nwords, http_act]
    else:
        return [r, url_len, url_depth, content_len, content_nwords]


def file_process(act, complete_ul=None, data='r4.2', filetype='act'):
    if filetype == 'act':
        ftype = act['url/fname'].split(".")[1]
        disk = 1 if act['url/fname'][0] == 'C' else 0
        if act['url/fname'][0] == 'R':
            disk = 2
        file_depth = act['url/fname'].count('\\')
    elif filetype == 'att':                                     # attachments
        tmp = act.split('.')[1]
        ftype = tmp[:tmp.find('(')]
        attsize = int(tmp[tmp.find("(") + 1:tmp.find(")")])
        r = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        if ftype in ['zip', 'rar', '7z']:
            ind = 1
        elif ftype in ['jpg', 'png', 'bmp']:
            ind = 2
        elif ftype in ['doc', 'docx', 'pdf']:
            ind = 3
        elif ftype in ['txt', 'cfg', 'rtf']:
            ind = 4
        elif ftype in ['exe', 'sh']:
            ind = 5
        else:
            ind = 0
        r[0][ind] = 1
        r[1][ind] = attsize
        return r

    fsize = len(act['content'])
    f_nwords = act['content'].count(' ') + 1
    if ftype in ['zip', 'rar', '7z']:
        r = 2
    elif ftype in ['jpg', 'png', 'bmp']:
        r = 3
    elif ftype in ['doc', 'docx', 'pdf']:
        r = 4
    elif ftype in ['txt', 'cfg', 'rtf']:
        r = 5
    elif ftype in ['exe', 'sh']:
        r = 6
    else:
        r = 1

    if data in R5 + R6:
        to_usb = 1 if act['to'] == 'True' else 0
        from_usb = 1 if act['from'] == 'True' else 0
        file_depth = act['url/fname'].count('\\')
        file_act_dict = {'file open': 1, 'file copy': 2,
                         'file write': 3, 'file delete': 4}
        file_act = file_act_dict.get(act['activity'].lower(), 0)
        return [r, fsize, f_nwords, disk, file_depth, file_act, to_usb, from_usb]
    elif data in R4:
        return [r, fsize, f_nwords, disk, file_depth]


def from_pc(act, ul):
    """Code 0,1,2,3: own pc, shared pc, other's pc, supervisor's pc."""
    user_pc = ul.loc[act['user']]['pc']
    act_pc = act['pc']
    if act_pc == user_pc:
        return (0, act_pc)
    elif (ul.loc[act['user']]['sharedpc'] is not None
          and act_pc in ul.loc[act['user']]['sharedpc']):
        return (1, act_pc)
    elif (ul.loc[act['user']]['sup'] is not None
          and act_pc == ul.loc[ul.loc[act['user']]['sup']]['pc']):
        return (3, act_pc)
    else:
        return (2, act_pc)


def get_sessions(uw, first_sid=0):
    """
    sessions[sid] = [sessionid, pc, start_with, end_with, start, end,
                     n_concurrent_login, [action_indices]]
      start_with: at the beginning of a week, starts with a log in or not (1, 2)
      end_with  : log off, or next log on same computer (1, 2)
    """
    sessions = {}
    open_sessions = {}
    sid = 0
    current_pc = uw.iloc[0]['pcid']
    start_time = uw.iloc[0]['time_stamp']
    if uw.iloc[0]['act'] == 1:
        open_sessions[current_pc] = [current_pc, 1, 0, start_time, start_time,
                                     1, [uw.index[0]]]
    else:
        open_sessions[current_pc] = [current_pc, 2, 0, start_time, start_time,
                                     1, [uw.index[0]]]

    for i in uw.index[1:]:
        current_pc = uw.loc[i]['pcid']
        if current_pc in open_sessions:
            if uw.loc[i]['act'] == 2:
                open_sessions[current_pc][2] = 1
                open_sessions[current_pc][4] = uw.loc[i]['time_stamp']
                open_sessions[current_pc][6].append(i)
                sessions[sid] = [first_sid + sid] + open_sessions.pop(current_pc)
                sid += 1
            elif uw.loc[i]['act'] == 1:
                open_sessions[current_pc][2] = 2
                sessions[sid] = [first_sid + sid] + open_sessions.pop(current_pc)
                sid += 1
                open_sessions[current_pc] = [current_pc, 1, 0, uw.loc[i]['time_stamp'],
                                             uw.loc[i]['time_stamp'], 1, [i]]
                if len(open_sessions) > 1:
                    for k in open_sessions:
                        open_sessions[k][5] += 1
            else:
                open_sessions[current_pc][4] = uw.loc[i]['time_stamp']
                open_sessions[current_pc][6].append(i)
        else:
            start_status = 1 if uw.loc[i]['act'] == 1 else 2
            open_sessions[current_pc] = [current_pc, start_status, 0,
                                         uw.loc[i]['time_stamp'],
                                         uw.loc[i]['time_stamp'], 1, [i]]
            if len(open_sessions) > 1:
                for k in open_sessions:
                    open_sessions[k][5] += 1
    return sessions


def get_u_features_dicts(ul, data='r5.2'):
    ufdict = {}
    list_uf = [] if data in R4 else ['project']
    list_uf += ['role', 'b_unit', 'f_unit', 'dept', 'team']
    for f in list_uf:
        # COMPAT: pandas 3.0 introduced a dedicated `str` dtype, and astype(str)
        # now PRESERVES missing values instead of rendering them as the literal
        # 'nan' that pandas 1.x/2.x produced. Some CERT LDAP fields are blank
        # for some users (commonly `team`), so the set below would then mix str
        # and float and tmp.sort() raises TypeError. Going via object dtype and
        # mapping str reproduces the 1.x behaviour exactly, including keeping
        # 'nan' as its own category label -- which matters, because that label
        # is baked into the encoded feature values of every published result.
        ul[f] = ul[f].astype(object).map(str)
        tmp = list(set(ul[f]))
        tmp.sort()
        ufdict[f] = {idx: i for i, idx in enumerate(tmp)}
    return (ul, ufdict, list_uf)


def proc_u_features(uf, ufdict, list_f=None, data='r4.2'):
    if type(list_f) != list:
        list_f = [] if data in R4 else ['project']
        list_f = ['role', 'b_unit', 'f_unit', 'dept', 'team'] + list_f
    out = []
    for f in list_f:
        out.append(ufdict[f][uf[f]])
    return out


def f_stats_calc(ud, fn, stats_f, countonly_f={}, get_stats=False):
    f_count = len(ud)
    r = []
    f_names = []
    for f in stats_f:
        inp = ud[f].values
        if get_stats:
            if f_count > 0:
                r += [np.min(inp), np.max(inp), np.median(inp),
                      np.mean(inp), np.std(inp)]
            else:
                r += [0, 0, 0, 0, 0]
            f_names += [fn + '_min_' + f, fn + '_max_' + f, fn + '_med_' + f,
                        fn + '_mean_' + f, fn + '_std_' + f]
        else:
            if f_count > 0:
                r += [np.mean(inp)]
            else:
                r += [0]
            f_names += [fn + '_mean_' + f]
    for f in countonly_f:
        for v in countonly_f[f]:
            r += [sum(ud[f].values == v)]
            f_names += [fn + '_n-' + f + str(v)]
    return (f_count, r, f_names)


def f_calc_subfeatures(ud, fname, filter_col, filter_vals, filter_names,
                       sub_features, countonly_subfeatures):
    [n, stats, fnames] = f_stats_calc(ud, fname, sub_features, countonly_subfeatures)
    allf = [n] + stats
    allf_names = ['n_' + fname] + fnames
    for i in range(len(filter_vals)):
        [n_sf, sf_stats, sf_fnames] = f_stats_calc(
            ud[ud[filter_col] == filter_vals[i]], filter_names[i],
            sub_features, countonly_subfeatures)
        allf += [n_sf] + sf_stats
        allf_names += ([fname + '_n_' + filter_names[i]]
                       + [fname + '_' + x for x in sf_fnames])
    return (allf, allf_names)


def f_calc(ud, mode='week', data='r4.2'):
    n_weekendact = (ud['time'] == 3).sum()
    is_weekend = 1 if n_weekendact > 0 else 0

    all_countonlyf = {'pc': [0, 1, 2, 3]} if mode != 'session' else {}
    [all_f, all_f_names] = f_calc_subfeatures(ud, 'allact', None, [], [], [],
                                              all_countonlyf)
    if mode == 'day':
        [workhourf, workhourf_names] = f_calc_subfeatures(
            ud[(ud['time'] == 1) | (ud['time'] == 3)], 'workhourallact',
            None, [], [], [], all_countonlyf)
        [afterhourf, afterhourf_names] = f_calc_subfeatures(
            ud[(ud['time'] == 2) | (ud['time'] == 4)], 'afterhourallact',
            None, [], [], [], all_countonlyf)
    elif mode == 'week':
        [workhourf, workhourf_names] = f_calc_subfeatures(
            ud[ud['time'] == 1], 'workhourallact', None, [], [], [], all_countonlyf)
        [afterhourf, afterhourf_names] = f_calc_subfeatures(
            ud[ud['time'] == 2], 'afterhourallact', None, [], [], [], all_countonlyf)
        [weekendf, weekendf_names] = f_calc_subfeatures(
            ud[ud['time'] >= 3], 'weekendallact', None, [], [], [], all_countonlyf)

    logon_countonlyf = {'pc': [0, 1, 2, 3]} if mode != 'session' else {}
    logon_statf = []
    [all_logonf, all_logonf_names] = f_calc_subfeatures(
        ud[ud['act'] == 1], 'logon', None, [], [], logon_statf, logon_countonlyf)
    if mode == 'day':
        [workhourlogonf, workhourlogonf_names] = f_calc_subfeatures(
            ud[(ud['act'] == 1) & ((ud['time'] == 1) | (ud['time'] == 3))],
            'workhourlogon', None, [], [], logon_statf, logon_countonlyf)
        [afterhourlogonf, afterhourlogonf_names] = f_calc_subfeatures(
            ud[(ud['act'] == 1) & ((ud['time'] == 2) | (ud['time'] == 4))],
            'afterhourlogon', None, [], [], logon_statf, logon_countonlyf)
    elif mode == 'week':
        [workhourlogonf, workhourlogonf_names] = f_calc_subfeatures(
            ud[(ud['act'] == 1) & (ud['time'] == 1)], 'workhourlogon',
            None, [], [], logon_statf, logon_countonlyf)
        [afterhourlogonf, afterhourlogonf_names] = f_calc_subfeatures(
            ud[(ud['act'] == 1) & (ud['time'] == 2)], 'afterhourlogon',
            None, [], [], logon_statf, logon_countonlyf)
        [weekendlogonf, weekendlogonf_names] = f_calc_subfeatures(
            ud[(ud['act'] == 1) & (ud['time'] >= 3)], 'weekendlogon',
            None, [], [], logon_statf, logon_countonlyf)

    device_countonlyf = {'pc': [0, 1, 2, 3]} if mode != 'session' else {}
    device_statf = ['usb_dur', 'file_tree_len'] if data not in R4 else ['usb_dur']
    [all_devicef, all_devicef_names] = f_calc_subfeatures(
        ud[ud['act'] == 3], 'usb', None, [], [], device_statf, device_countonlyf)
    if mode == 'day':
        [workhourdevicef, workhourdevicef_names] = f_calc_subfeatures(
            ud[(ud['act'] == 3) & ((ud['time'] == 1) | (ud['time'] == 3))],
            'workhourusb', None, [], [], device_statf, device_countonlyf)
        [afterhourdevicef, afterhourdevicef_names] = f_calc_subfeatures(
            ud[(ud['act'] == 3) & ((ud['time'] == 2) | (ud['time'] == 4))],
            'afterhourusb', None, [], [], device_statf, device_countonlyf)
    elif mode == 'week':
        [workhourdevicef, workhourdevicef_names] = f_calc_subfeatures(
            ud[(ud['act'] == 3) & (ud['time'] == 1)], 'workhourusb',
            None, [], [], device_statf, device_countonlyf)
        [afterhourdevicef, afterhourdevicef_names] = f_calc_subfeatures(
            ud[(ud['act'] == 3) & (ud['time'] == 2)], 'afterhourusb',
            None, [], [], device_statf, device_countonlyf)
        [weekenddevicef, weekenddevicef_names] = f_calc_subfeatures(
            ud[(ud['act'] == 3) & (ud['time'] >= 3)], 'weekendusb',
            None, [], [], device_statf, device_countonlyf)

    if mode != 'session':
        file_countonlyf = {'to_usb': [1], 'from_usb': [1], 'file_act': [1, 2, 3, 4],
                           'disk': [0, 1], 'pc': [0, 1, 2, 3]}
    else:
        file_countonlyf = {'to_usb': [1], 'from_usb': [1], 'file_act': [1, 2, 3, 4],
                           'disk': [0, 1, 2]}
    if data in R4:
        [file_countonlyf.pop(k) for k in ['to_usb', 'from_usb', 'file_act']]

    _ftypes = [1, 2, 3, 4, 5, 6]
    _fnames = ['otherf', 'compf', 'phof', 'docf', 'txtf', 'exef']
    _fstats = ['file_len', 'file_depth', 'file_nwords']
    (all_filef, all_filef_names) = f_calc_subfeatures(
        ud[ud['act'] == 7], 'file', 'file_type', _ftypes, _fnames,
        _fstats, file_countonlyf)
    if mode == 'day':
        (workhourfilef, workhourfilef_names) = f_calc_subfeatures(
            ud[(ud['act'] == 7) & ((ud['time'] == 1) | (ud['time'] == 3))],
            'workhourfile', 'file_type', _ftypes, _fnames, _fstats, file_countonlyf)
        (afterhourfilef, afterhourfilef_names) = f_calc_subfeatures(
            ud[(ud['act'] == 7) & ((ud['time'] == 2) | (ud['time'] == 4))],
            'afterhourfile', 'file_type', _ftypes, _fnames, _fstats, file_countonlyf)
    elif mode == 'week':
        (workhourfilef, workhourfilef_names) = f_calc_subfeatures(
            ud[(ud['act'] == 7) & (ud['time'] == 1)], 'workhourfile',
            'file_type', _ftypes, _fnames, _fstats, file_countonlyf)
        (afterhourfilef, afterhourfilef_names) = f_calc_subfeatures(
            ud[(ud['act'] == 7) & (ud['time'] == 2)], 'afterhourfile',
            'file_type', _ftypes, _fnames, _fstats, file_countonlyf)
        (weekendfilef, weekendfilef_names) = f_calc_subfeatures(
            ud[(ud['act'] == 7) & (ud['time'] >= 3)], 'weekendfile',
            'file_type', _ftypes, _fnames, _fstats, file_countonlyf)

    email_stats_f = ['n_des', 'n_atts', 'n_exdes', 'n_bccdes', 'email_size',
                     'email_text_slen', 'email_text_nwords']
    if data not in R4:
        email_stats_f += ['e_att_other', 'e_att_comp', 'e_att_pho', 'e_att_doc',
                          'e_att_txt', 'e_att_exe']
        email_stats_f += ['e_att_sother', 'e_att_scomp', 'e_att_spho',
                          'e_att_sdoc', 'e_att_stxt', 'e_att_sexe']
        mail_filter = 'send_mail'
        mail_filter_vals = [0, 1]
        mail_filter_names = ['recvmail', 'send_mail']
    else:
        mail_filter, mail_filter_vals, mail_filter_names = None, [], []

    if mode != 'session':
        mail_countonlyf = {'Xemail': [1], 'exbccmail': [1], 'pc': [0, 1, 2, 3]}
    else:
        mail_countonlyf = {'Xemail': [1], 'exbccmail': [1]}

    (all_emailf, all_emailf_names) = f_calc_subfeatures(
        ud[ud['act'] == 6], 'email', mail_filter, mail_filter_vals,
        mail_filter_names, email_stats_f, mail_countonlyf)
    if mode == 'week':
        (workhouremailf, workhouremailf_names) = f_calc_subfeatures(
            ud[(ud['act'] == 6) & (ud['time'] == 1)], 'workhouremail',
            mail_filter, mail_filter_vals, mail_filter_names,
            email_stats_f, mail_countonlyf)
        (afterhouremailf, afterhouremailf_names) = f_calc_subfeatures(
            ud[(ud['act'] == 6) & (ud['time'] == 2)], 'afterhouremail',
            mail_filter, mail_filter_vals, mail_filter_names,
            email_stats_f, mail_countonlyf)
        (weekendemailf, weekendemailf_names) = f_calc_subfeatures(
            ud[(ud['act'] == 6) & (ud['time'] >= 3)], 'weekendemail',
            mail_filter, mail_filter_vals, mail_filter_names,
            email_stats_f, mail_countonlyf)
    elif mode == 'day':
        (workhouremailf, workhouremailf_names) = f_calc_subfeatures(
            ud[(ud['act'] == 6) & ((ud['time'] == 1) | (ud['time'] == 3))],
            'workhouremail', mail_filter, mail_filter_vals, mail_filter_names,
            email_stats_f, mail_countonlyf)
        (afterhouremailf, afterhouremailf_names) = f_calc_subfeatures(
            ud[(ud['act'] == 6) & ((ud['time'] == 2) | (ud['time'] == 4))],
            'afterhouremail', mail_filter, mail_filter_vals, mail_filter_names,
            email_stats_f, mail_countonlyf)

    if data in R5 or data in R4:
        http_count_subf = {'pc': [0, 1, 2, 3]}
    elif data in R6:
        http_count_subf = {'pc': [0, 1, 2, 3], 'http_act': [1, 2, 3]}
    if mode == 'session':
        http_count_subf.pop('pc', None)

    _htypes = [1, 2, 3, 4, 5, 6]
    _hnames = ['otherf', 'socnetf', 'cloudf', 'jobf', 'leakf', 'hackf']
    _hstats = ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords']
    (all_httpf, all_httpf_names) = f_calc_subfeatures(
        ud[ud['act'] == 5], 'http', 'http_type', _htypes, _hnames,
        _hstats, http_count_subf)
    if mode == 'week':
        (workhourhttpf, workhourhttpf_names) = f_calc_subfeatures(
            ud[(ud['act'] == 5) & (ud['time'] == 1)], 'workhourhttp',
            'http_type', _htypes, _hnames, _hstats, http_count_subf)
        (afterhourhttpf, afterhourhttpf_names) = f_calc_subfeatures(
            ud[(ud['act'] == 5) & (ud['time'] == 2)], 'afterhourhttp',
            'http_type', _htypes, _hnames, _hstats, http_count_subf)
        (weekendhttpf, weekendhttpf_names) = f_calc_subfeatures(
            ud[(ud['act'] == 5) & (ud['time'] >= 3)], 'weekendhttp',
            'http_type', _htypes, _hnames, _hstats, http_count_subf)
    elif mode == 'day':
        (workhourhttpf, workhourhttpf_names) = f_calc_subfeatures(
            ud[(ud['act'] == 5) & ((ud['time'] == 1) | (ud['time'] == 3))],
            'workhourhttp', 'http_type', _htypes, _hnames, _hstats, http_count_subf)
        (afterhourhttpf, afterhourhttpf_names) = f_calc_subfeatures(
            ud[(ud['act'] == 5) & ((ud['time'] == 2) | (ud['time'] == 4))],
            'afterhourhttp', 'http_type', _htypes, _hnames, _hstats, http_count_subf)

    numActs = all_f[0]
    mal_u = 0
    if (ud['mal_act']).sum() > 0:
        tmp = list(set(ud['insider']))
        if len(tmp) > 1:
            tmp.remove(0.0)
        mal_u = tmp[0]

    if mode == 'week':
        features_tmp = (all_f + workhourf + afterhourf + weekendf
                        + all_logonf + workhourlogonf + afterhourlogonf + weekendlogonf
                        + all_devicef + workhourdevicef + afterhourdevicef + weekenddevicef
                        + all_filef + workhourfilef + afterhourfilef + weekendfilef
                        + all_emailf + workhouremailf + afterhouremailf + weekendemailf
                        + all_httpf + workhourhttpf + afterhourhttpf + weekendhttpf)
        fnames_tmp = (all_f_names + workhourf_names + afterhourf_names + weekendf_names
                      + all_logonf_names + workhourlogonf_names + afterhourlogonf_names
                      + weekendlogonf_names
                      + all_devicef_names + workhourdevicef_names
                      + afterhourdevicef_names + weekenddevicef_names
                      + all_filef_names + workhourfilef_names + afterhourfilef_names
                      + weekendfilef_names
                      + all_emailf_names + workhouremailf_names
                      + afterhouremailf_names + weekendemailf_names
                      + all_httpf_names + workhourhttpf_names
                      + afterhourhttpf_names + weekendhttpf_names)
    elif mode == 'day':
        features_tmp = (all_f + workhourf + afterhourf
                        + all_logonf + workhourlogonf + afterhourlogonf
                        + all_devicef + workhourdevicef + afterhourdevicef
                        + all_filef + workhourfilef + afterhourfilef
                        + all_emailf + workhouremailf + afterhouremailf
                        + all_httpf + workhourhttpf + afterhourhttpf)
        fnames_tmp = (all_f_names + workhourf_names + afterhourf_names
                      + all_logonf_names + workhourlogonf_names + afterhourlogonf_names
                      + all_devicef_names + workhourdevicef_names + afterhourdevicef_names
                      + all_filef_names + workhourfilef_names + afterhourfilef_names
                      + all_emailf_names + workhouremailf_names + afterhouremailf_names
                      + all_httpf_names + workhourhttpf_names + afterhourhttpf_names)
    elif mode == 'session':
        features_tmp = (all_f + all_logonf + all_devicef + all_filef
                        + all_emailf + all_httpf)
        fnames_tmp = (all_f_names + all_logonf_names + all_devicef_names
                      + all_filef_names + all_emailf_names + all_httpf_names)

    return [numActs, is_weekend, features_tmp, fnames_tmp, mal_u]


def session_instance_calc(ud, sinfo, week, mode, data, uw, v, list_uf):
    d = ud.iloc[0]['day']
    perworkhour = sum(ud['time'] == 1) / len(ud)
    perafterhour = sum(ud['time'] == 2) / len(ud)
    perweekend = sum(ud['time'] == 3) / len(ud)
    perweekendafterhour = sum(ud['time'] == 4) / len(ud)
    st_timestamp = min(ud['time_stamp'])
    end_timestamp = max(ud['time_stamp'])
    s_dur = (end_timestamp - st_timestamp).total_seconds() / 60          # minutes
    s_start = st_timestamp.hour + st_timestamp.minute / 60
    s_end = end_timestamp.hour + end_timestamp.minute / 60
    starttime = st_timestamp.timestamp()
    endtime = end_timestamp.timestamp()
    n_days = len(set(ud['day']))
    tmp = f_calc(ud, mode, data)
    session_instance = ([starttime, endtime, v, sinfo[0], d, week,
                         ud.iloc[0]['pc'], perworkhour, perafterhour, perweekend,
                         perweekendafterhour, n_days, s_dur, sinfo[6], sinfo[2],
                         sinfo[3], s_start, s_end]
                        + (uw.loc[v, list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N']]).tolist()
                        + tmp[2] + [tmp[4]])
    return (session_instance, tmp[3])


# ══════════════════════════════════════════════════════════════════════════════
#
#  PART B -- ORCHESTRATION  (rewritten for speed, resumability, observability)
#
#  Nothing here decides WHAT a feature is; it only decides how work is scheduled,
#  cached and recorded.
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 1 ────────────────────────────────────────────────────────────────────

def _column_spec(act: str, dname: str):
    """Per-release CSV column layout. Verbatim from the original."""
    if act == 'email':
        if dname in R4:
            return ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from',
                    'size', '#att', 'content']
        return ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from',
                'activity', 'size', 'att', 'content']
    if act == 'logon':
        return ['id', 'date', 'user', 'pc', 'activity']
    if act == 'device':
        if dname in R4:
            return ['id', 'date', 'user', 'pc', 'activity']
        return ['id', 'date', 'user', 'pc', 'content', 'activity']
    if act == 'http':
        if dname in R6:
            return ['id', 'date', 'user', 'pc', 'url/fname', 'activity', 'content']
        return ['id', 'date', 'user', 'pc', 'url/fname', 'content']
    if act == 'file':
        if dname in R4:
            return ['id', 'date', 'user', 'pc', 'url/fname', 'content']
        return ['id', 'date', 'user', 'pc', 'url/fname', 'activity', 'to',
                'from', 'content']
    raise ValueError(act)


class _WeekResolver:
    """
    Map a 'MM/DD/YYYY HH:MM:SS' string to its week index.

    The original called datetime.strptime() for this on every log line. The week
    index depends only on the date portion, and there are ~516 distinct dates in
    a release, so a dict keyed on the first 10 characters collapses ~135 million
    parses (r6.2) into a few hundred. Results are bit-identical.
    """

    __slots__ = ("_cache", "_startdate")

    def __init__(self, first_date: str):
        self._cache = {}
        self._startdate = datetime.strptime(first_date, '%Y-%m-%d')

    def week_of(self, timestamp: str) -> int:
        key = timestamp[:10]                       # 'MM/DD/YYYY'
        wk = self._cache.get(key)
        if wk is None:
            dt = datetime.strptime(key, '%m/%d/%Y')
            wk = (dt - self._startdate).days // 7
            self._cache[key] = wk
        return wk


def _read_first_data_line(path: Path) -> str:
    """First non-header line of a CSV, without shelling out to `head`."""
    with open(path, 'r', errors='replace') as fh:
        fh.readline()                              # header
        return fh.readline()


def _add_action_thisweek(act, columns, lines, act_handles, week_index, stop,
                         resolver: _WeekResolver, dname='r5.2'):
    """
    Collect one activity file's rows for the current week.

    Structurally identical to the original `add_action_thisweek`. Two changes:
      * the week test uses the cached resolver instead of a fresh strptime;
      * the DataFrame is built once from the accumulated list (unchanged), and
        the caller concatenates rather than using DataFrame.append.

    The raw-line split is deliberately preserved, INCLUDING the fact that the
    final field retains its trailing newline. Step 3 compares device activity
    against the literal 'Connect\\n' / 'Disconnect\\n', so stripping it here
    would silently change every usb_dur value.
    """
    thisweek_act = []
    while True:
        if not lines[act]:
            stop[act] = 1
            break
        raw = lines[act]
        if dname in R6 and act in ['email', 'file', 'http'] and '"' in raw:
            firstpart = raw[:raw.find('"') - 1]
            content = raw[raw.find('"') + 1:-1]
            tmp = firstpart.split(',') + [content]
        else:
            tmp = raw.split(',')

        if resolver.week_of(tmp[1]) == week_index:
            thisweek_act.append(tmp)
        else:
            break
        lines[act] = act_handles[act].readline()

    df = pd.DataFrame(thisweek_act, columns=columns)
    df['type'] = act
    df.index = df['id']
    df = df.drop(columns='id')                     # COMPAT: positional axis arg gone
    return df


def step1_split_by_week(dname: str, week_dir: Path, ckpt: CheckpointDB,
                        progress: ProgressReporter, resume: bool):
    """
    Merge the five activity logs into one pickle per week.

    Inherently sequential: each source file is time-ordered and is streamed
    once, so this cannot be parallelised without re-sorting. It is instead made
    cheap (see _WeekResolver) and resumable at week granularity.
    """
    week_dir.mkdir(parents=True, exist_ok=True)
    done = ckpt.completed("step1", "-") if resume else set()
    n_weeks = NUM_WEEKS[dname]

    if len(done) >= n_weeks:
        log.info("Step 1: all %d weekly partitions already present -- skipping", n_weeks)
        return

    allacts = ['device', 'email', 'file', 'http', 'logon']

    # Establish the reference start date exactly as the original did: the first
    # http timestamp, rolled back to the preceding Sunday.
    firstline = _read_first_data_line(Path('http.csv'))
    firstdate = time_convert(firstline.split(',')[1], 't2dt')
    firstdate = firstdate - timedelta(int(firstdate.strftime("%w")))
    firstdate = time_convert(firstdate, 'dt2date')
    resolver = _WeekResolver(firstdate)

    act_handles, lines, stop = {}, {}, {}
    for act in allacts:
        act_handles[act] = open(act + '.csv', 'r')
        next(act_handles[act], None)               # skip header
        lines[act] = act_handles[act].readline()
        stop[act] = 0

    progress.add("step1", "Step 1  weekly split", n_weeks)
    week_index = 0
    try:
        while sum(stop.values()) < 5:
            t0 = time.time()
            target = week_dir / f"{week_index}.pickle"

            # Even when a week is already done we must still advance the file
            # handles, or the stream position desynchronises from week_index.
            frames = []
            for act in allacts:
                columns = _column_spec(act, dname)
                frames.append(_add_action_thisweek(act, columns, lines, act_handles,
                                                   week_index, stop, resolver,
                                                   dname=dname))

            if week_index in done and target.is_file():
                week_index += 1
                progress.advance("step1", 1)
                continue

            # COMPAT + speed: DataFrame.append is removed in pandas 2.0 and was
            # quadratic besides. One concat over five frames instead.
            thisweekdf = pd.concat(frames, sort=False) if frames else pd.DataFrame()
            if len(thisweekdf):
                # Vectorised parse: one pass in C rather than one strptime per row.
                thisweekdf['date'] = pd.to_datetime(thisweekdf['date'],
                                                    format="%m/%d/%Y %H:%M:%S")

            ckpt.mark("step1", "-", week_index, "running")
            rows = atomic_write_pickle(thisweekdf, target)
            ckpt.mark("step1", "-", week_index, "completed",
                      duration=time.time() - t0, rows=rows, output_path=str(target))

            week_index += 1
            progress.advance("step1", 1, rate_hint=f"{rows:,} rows")
    finally:
        for fh in act_handles.values():
            fh.close()


# ── Step 3 ────────────────────────────────────────────────────────────────────

def _connect_duration(pos, row_date, pc, disc_pos, conn_pos, dates):
    """
    USB connect duration, replicating the original semantics precisely.

    Original:
        tmp = df_acts_u.iloc[pos:]                      # note: INCLUSIVE of pos
        disconnect_acts = tmp[activity=='Disconnect\\n' & pc matches]
        connect_acts    = tmp[activity=='Connect\\n'    & pc matches]
        if disconnect_acts: distime = first one's date
            if connect_acts and first connect date < distime: -1
            else: seconds between row_date and distime
        else: -1

    Because the window starts AT `pos`, `connect_acts` normally contains the
    current Connect action itself, whose date precedes the disconnect -- so the
    original very often returns -1 here. That is preserved deliberately: this
    function must reproduce the published feature values, not repair them.

    The original re-scanned the user's remaining actions for every Connect,
    which is quadratic. Here the per-PC position arrays are precomputed once
    and searched with np.searchsorted.
    """
    d_arr = disc_pos.get(pc)
    if d_arr is None or len(d_arr) == 0:
        return -1
    di = np.searchsorted(d_arr, pos, side='left')
    if di >= len(d_arr):
        return -1
    distime = dates[d_arr[di]]

    c_arr = conn_pos.get(pc)
    if c_arr is not None and len(c_arr):
        ci = np.searchsorted(c_arr, pos, side='left')
        if ci < len(c_arr) and dates[c_arr[ci]] < distime:
            return -1

    td = distime - row_date
    return td.days * 24 * 3600 + td.seconds


def process_week_num(week, users, num_dir, userlist='all', data='r4.2',
                     week_dir="DataByWeek"):
    """
    Convert every action in a week to its numeric feature row.

    Same output as the original `process_week_num`. Rewritten hot path:
      * one groupby instead of a full scan per user;
      * rows materialised as dicts once instead of a Series per access;
      * per-PC connect/disconnect indices instead of a tail rescan per Connect.
    """
    user_dict = {idx: i for (i, idx) in enumerate(users.index)}

    # A release may contain fewer populated weeks than NUM_WEEKS (truncated or
    # partial data). An absent or empty week is not an error -- it just yields
    # no instances -- so record it as complete rather than failing the run.
    src = Path(week_dir) / f"{week}.pickle"
    if not src.is_file():
        log.debug("Step 3 week %d: no source partition, skipping", week)
        return 0
    acts_week = pd.read_pickle(src)
    if len(acts_week) == 0:
        return 0

    start_week, end_week = min(acts_week.date), max(acts_week.date)
    acts_week = acts_week.sort_values('date', ascending=True)

    n_cols = 45 if data in R5 else 46
    if data in R4:
        n_cols = 27

    u_week = np.zeros((len(acts_week), n_cols))
    pc_time = []

    uacts_mapping = {'logon': 1, 'logoff': 2, 'connect': 3, 'disconnect': 4,
                     'http': 5, 'email': 6, 'file': 7}

    grouped = acts_week.groupby('user', sort=False)
    wanted = None if userlist == 'all' else set(userlist)

    current_ind = 0
    for u, df_acts_u in grouped:
        if wanted is not None and u not in wanted:
            continue

        mal_u = 0
        if users.loc[u].malscene > 0:
            if start_week <= users.loc[u].mend and users.loc[u].mstart <= end_week:
                mal_u = users.loc[u].malscene

        # Materialise once. Every downstream feature function takes a mapping,
        # so dicts are a drop-in for the original .iloc[i] Series.
        rows = df_acts_u.to_dict('records')
        idx_list = list(df_acts_u.index)
        n_u = len(rows)

        list_uacts = df_acts_u.type.tolist()
        list_activity = df_acts_u.activity.tolist()
        list_uacts = [list_activity[i].strip().lower()
                      if (type(list_activity[i]) == str
                          and list_activity[i].strip() in
                          ['Logon', 'Logoff', 'Connect', 'Disconnect'])
                      else list_uacts[i] for i in range(len(list_uacts))]
        list_uacts_num = [uacts_mapping[x] for x in list_uacts]

        # Precompute Connect/Disconnect positions per PC for this user.
        disc_pos, conn_pos = defaultdict(list), defaultdict(list)
        for i in range(n_u):
            a = rows[i].get('activity')
            if a == 'Disconnect\n':
                disc_pos[rows[i]['pc']].append(i)
            elif a == 'Connect\n':
                conn_pos[rows[i]['pc']].append(i)
        disc_pos = {k: np.asarray(v) for k, v in disc_pos.items()}
        conn_pos = {k: np.asarray(v) for k, v in conn_pos.items()}
        dates = [r['date'] for r in rows]

        # The original tested `idx in users.loc[u]['malacts']` directly against
        # the numpy array -- O(n) per action. A set makes it O(1), but pandas
        # may have stored a single-element assignment as a 0-d array, which is
        # not iterable; atleast_1d normalises both cases.
        malacts_u = users.loc[u]['malacts'] if mal_u > 0 else None
        if malacts_u is None:
            malacts_set = set()
        else:
            malacts_set = set(np.atleast_1d(malacts_u).tolist())

        oneu_week = np.zeros((n_u, n_cols))
        oneu_pc_time = []

        for i in range(n_u):
            act = rows[i]
            pc, _ = from_pc(act, users)

            if is_weekend(act['date']):
                act_time = 4 if is_after_whour(act['date']) else 3
            elif is_after_whour(act['date']):
                act_time = 2
            else:
                act_time = 1

            if data in R4:
                device_f = [0]
                file_f = [0, 0, 0, 0, 0]
                http_f = [0, 0, 0, 0, 0]
                email_f = [0] * 9
            else:
                device_f = [0, 0]
                file_f = [0] * 8
                http_f = [0, 0, 0, 0, 0]
                if data in R6:
                    http_f = [0, 0, 0, 0, 0, 0]
                email_f = [0] * 23

            kind = list_uacts[i]
            if kind == 'file':
                file_f = file_process(act, data=data)
            elif kind == 'email':
                email_f = email_process(act, data=data)
            elif kind == 'http':
                http_f = http_process(act, data=data)
            elif kind == 'connect':
                connect_dur = _connect_duration(i, act['date'], act['pc'],
                                                disc_pos, conn_pos, dates)
                if data in R5 + R6:
                    file_tree_len = len(act['content'].split(';'))
                    device_f = [connect_dur, file_tree_len]
                else:
                    device_f = [connect_dur]

            is_mal_act = 1 if (mal_u > 0 and idx_list[i] in malacts_set) else 0

            oneu_week[i, :] = ([user_dict[u],
                                time_convert(act['date'], 'dt2dn'),
                                list_uacts_num[i], pc, act_time]
                               + device_f + file_f + http_f + email_f
                               + [is_mal_act, mal_u])
            oneu_pc_time.append([idx_list[i], act['pc'], act['date']])

        u_week[current_ind:current_ind + n_u, :] = oneu_week
        pc_time += oneu_pc_time
        current_ind += n_u

    u_week = u_week[0:current_ind, :]

    col_names = ['user', 'day', 'act', 'pc', 'time']
    if data in R4:
        device_feature_names = ['usb_dur']
        file_feature_names = ['file_type', 'file_len', 'file_nwords', 'disk',
                              'file_depth']
        http_feature_names = ['http_type', 'url_len', 'url_depth', 'http_c_len',
                              'http_c_nwords']
        email_feature_names = ['n_des', 'n_atts', 'Xemail', 'n_exdes', 'n_bccdes',
                               'exbccmail', 'email_size', 'email_text_slen',
                               'email_text_nwords']
    else:
        device_feature_names = ['usb_dur', 'file_tree_len']
        file_feature_names = ['file_type', 'file_len', 'file_nwords', 'disk',
                              'file_depth', 'file_act', 'to_usb', 'from_usb']
        http_feature_names = ['http_type', 'url_len', 'url_depth', 'http_c_len',
                              'http_c_nwords']
        if data in R6:
            http_feature_names = ['http_type', 'url_len', 'url_depth',
                                  'http_c_len', 'http_c_nwords', 'http_act']
        email_feature_names = ['send_mail', 'receive_mail', 'n_des', 'n_atts',
                               'Xemail', 'n_exdes', 'n_bccdes', 'exbccmail',
                               'email_size', 'email_text_slen', 'email_text_nwords']
        email_feature_names += ['e_att_other', 'e_att_comp', 'e_att_pho',
                                'e_att_doc', 'e_att_txt', 'e_att_exe']
        email_feature_names += ['e_att_sother', 'e_att_scomp', 'e_att_spho',
                                'e_att_sdoc', 'e_att_stxt', 'e_att_sexe']

    col_names = (col_names + device_feature_names + file_feature_names
                 + http_feature_names + email_feature_names + ['mal_act', 'insider'])

    df_u_week = pd.DataFrame(columns=['actid', 'pcid', 'time_stamp'] + col_names,
                             index=np.arange(0, len(pc_time)))
    df_u_week[['actid', 'pcid', 'time_stamp']] = np.array(pc_time, dtype=object)
    df_u_week[col_names] = u_week
    df_u_week[col_names] = df_u_week[col_names].astype(int)

    return atomic_write_pickle(df_u_week, Path(num_dir) / f"{week}_num.pickle")


# ── Step 4 ────────────────────────────────────────────────────────────────────

def to_csv(week, mode, data, ul, uf_dict, list_uf, num_dir, tmp_dir,
           subsession_mode={}):
    """
    Aggregate numeric action rows into week / day / session / subsession
    instances and write one pickle per (week, mode).

    Same output as the original `to_csv`. Rewritten hot path: the original
    re-scanned the whole week with `w[w['user']==v]` for every user (twice --
    once to build the metadata row, once to extract instances) and then
    `uactw[uactw['day']==d]` for every day. Both are now single groupby passes.
    """
    user_dict = {i: idx for (i, idx) in enumerate(ul.index)}

    if mode == 'session':
        # First 1-2 digits of the index encode the week, as in the original.
        first_sid = week * 100000
        cols2a = ['starttime', 'endtime', 'user', 'sessionid', 'day', 'week', 'pc',
                  'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour',
                  'n_days', 'duration', 'n_concurrent_sessions', 'start_with',
                  'end_with', 'ses_start', 'ses_end'] + list_uf + \
                 ['ITAdmin', 'O', 'C', 'E', 'A', 'N']
    elif mode == 'day':
        cols2a = ['starttime', 'endtime', 'user', 'day', 'week', 'isweekday',
                  'isweekend'] + list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N']
    else:
        cols2a = ['starttime', 'endtime', 'user', 'week'] + list_uf + \
                 ['ITAdmin', 'O', 'C', 'E', 'A', 'N']
    cols2b = ['insider']

    src = Path(num_dir) / f"{week}_num.pickle"
    if not src.is_file():
        log.debug("Step 4 [%s] week %d: no numeric partition, skipping", mode, week)
        return 0
    w = pd.read_pickle(src)
    if len(w) == 0:
        return 0

    # One grouping pass, reused for both the metadata table and the instances.
    by_user = {int(uid): sub for uid, sub in w.groupby('user', sort=False)}
    usnlist = list(by_user.keys())

    cols = ['week'] + list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N', 'insider']
    uwdict = {}
    for v in usnlist:
        uname = user_dict[v]
        is_ITAdmin = 1 if ul.loc[uname, 'role'] == 'ITAdmin' else 0
        row = ([week] + proc_u_features(ul.loc[uname], uf_dict, list_uf, data=data)
               + [is_ITAdmin] + (ul.loc[uname, ['O', 'C', 'E', 'A', 'N']]).tolist()
               + [0])
        row[-1] = int(list(set(by_user[v]['insider']))[0])
        uwdict[v] = row
    uw = pd.DataFrame.from_dict(uwdict, orient='index', columns=cols)

    # Precompute each user's metadata suffix once instead of a .loc per instance.
    meta_cols = list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N']
    meta_of = {v: uw.loc[v, meta_cols].tolist() for v in usnlist}

    towrite_list = []
    i_fnames = None
    if mode == 'session' and len(subsession_mode) > 0:
        towrite_list_subsession = {k1: {k2: [] for k2 in subsession_mode[k1]}
                                   for k1 in subsession_mode}

    for v in usnlist:
        uactw = by_user[v]

        if mode == 'week':
            a = uactw.iloc[0]['time_stamp']
            a = a - timedelta(int(a.strftime("%w")))          # nearest Sunday
            starttime = datetime(a.year, a.month, a.day).timestamp()
            endtime = (datetime(a.year, a.month, a.day) + timedelta(days=7)).timestamp()
            if len(uactw) > 0:
                tmp = f_calc(uactw, mode, data)
                i_fnames = tmp[3]
                towrite_list.append([starttime, endtime, v, week]
                                    + meta_of[v] + tmp[2] + [tmp[4]])

        elif mode == 'session':
            sessions = get_sessions(uactw, first_sid)
            first_sid += len(sessions)
            for s in sessions:
                sinfo = sessions[s]
                ud = uactw.loc[sessions[s][7]]
                if len(ud) == 0:
                    continue
                session_instance, i_fnames = session_instance_calc(
                    ud, sinfo, week, mode, data, uw, v, list_uf)
                towrite_list.append(session_instance)

                # Subsessions by consecutive time chunks
                if 'time' in subsession_mode:
                    for subsession_dur in subsession_mode['time']:
                        n_sub = int(np.ceil(session_instance[12] / subsession_dur))
                        if n_sub == 1:
                            towrite_list_subsession['time'][subsession_dur].append(
                                [0] + session_instance)
                        else:
                            sinfo1 = sinfo.copy()
                            for k in range(n_sub):
                                sinfo1[3] = 0 if k < n_sub - 1 else sinfo[3]
                                lo = sessions[s][4] + timedelta(minutes=k * subsession_dur)
                                hi = sessions[s][4] + timedelta(minutes=(k + 1) * subsession_dur)
                                sub_ud = ud[(ud['time_stamp'] >= lo)
                                            & (ud['time_stamp'] < hi)]
                                if len(sub_ud) > 0:
                                    ss_instance, _ = session_instance_calc(
                                        sub_ud, sinfo1, week, mode, data, uw, v, list_uf)
                                    towrite_list_subsession['time'][subsession_dur].append(
                                        [k] + ss_instance)

                # Subsessions by action count
                if 'nact' in subsession_mode:
                    for ss_nact in subsession_mode['nact']:
                        n_sub = int(np.ceil(len(ud) / ss_nact))
                        if n_sub == 1:
                            towrite_list_subsession['nact'][ss_nact].append(
                                [0] + session_instance)
                        else:
                            sinfo1 = sinfo.copy()
                            for k in range(n_sub):
                                sinfo1[3] = 0 if k < n_sub - 1 else sinfo[3]
                                ss_ud = ud.iloc[k * ss_nact:
                                                min(len(ud), (k + 1) * ss_nact)]
                                if len(ss_ud) > 0:
                                    ss_instance, _ = session_instance_calc(
                                        ss_ud, sinfo1, week, mode, data, uw, v, list_uf)
                                    towrite_list_subsession['nact'][ss_nact].append(
                                        [k] + ss_instance)

        elif mode == 'day':
            for d, ud in uactw.groupby('day', sort=True):
                isweekday = 1 if sum(ud['time'] >= 3) == 0 else 0
                isweekend = 1 - isweekday
                a = ud.iloc[0]['time_stamp']
                starttime = datetime(a.year, a.month, a.day).timestamp()
                endtime = (datetime(a.year, a.month, a.day)
                           + timedelta(days=1)).timestamp()
                if len(ud) > 0:
                    tmp = f_calc(ud, mode, data)
                    i_fnames = tmp[3]
                    towrite_list.append([starttime, endtime, v, d, week,
                                         isweekday, isweekend]
                                        + meta_of[v] + tmp[2] + [tmp[4]])

    if i_fnames is None:
        i_fnames = []

    towrite = pd.DataFrame(columns=cols2a + i_fnames + cols2b, data=towrite_list)
    rows = atomic_write_pickle(towrite, Path(tmp_dir) / f"{week}{mode}.pickle")

    if mode == 'session' and len(subsession_mode) > 0:
        for k1 in subsession_mode:
            for k2 in subsession_mode[k1]:
                df_tmp = pd.DataFrame(columns=['subs_ind'] + cols2a + i_fnames + cols2b,
                                      data=towrite_list_subsession[k1][k2])
                atomic_write_pickle(df_tmp,
                                    Path(tmp_dir) / f"{week}{mode}{k1}{k2}.pickle")
    return rows


# ── Consolidation ─────────────────────────────────────────────────────────────

def consolidate(week_range, mode, dname, tmp_dir: Path, out_dir: Path,
                suffix: str = "", drop_informational: bool = False) -> Path:
    """
    Stream per-week pickles into a single CSV.

    Written incrementally so peak memory stays at one week rather than the whole
    release, and written to .tmp then renamed so a partial CSV is never mistaken
    for a finished one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{mode}{suffix}{dname}.csv"
    tmp = target.with_suffix(".csv.tmp")

    header_written = False
    missing = 0
    with open(tmp, 'w', newline='') as fh:
        for w in week_range:
            src = tmp_dir / f"{w}{mode}{suffix}.pickle"
            if not src.is_file():
                # A week with no activity produces no pickle. Genuine failures
                # are already logged loudly by the Step 4 driver and recorded in
                # the checkpoint DB, so this is a count, not a warning per week.
                missing += 1
                log.debug("consolidate: no partition for week %d (%s%s)",
                          w, mode, suffix)
                continue
            df = pd.read_pickle(src)
            if drop_informational:
                drop = [c for c in INFORMATIONAL_FIELDS if c in df.columns]
                df = df.drop(columns=drop)
            df.to_csv(fh, header=not header_written, index=False)
            header_written = True

    if missing:
        log.info("  %s%s: %d/%d weeks had no data",
                 mode, suffix, missing, len(week_range))
    os.replace(tmp, target)
    return target


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(dname, users, ul, uf_dict, list_uf, num_dir, tmp_dir,
                  subsession_mode, worker_grid, backend="loky", timeout=1800,
                  n_partitions=12):
    """
    Time Step 4 on small / median / large partitions across worker counts.

    Primary metric is wall-clock to produce valid output -- not CPU utilisation,
    which rewards oversubscription that does not actually finish sooner.

    Each trial is bounded by `timeout`. Without it, a worker that dies mid-task
    leaves joblib waiting in _retrieve indefinitely, and the benchmark -- whose
    entire purpose is to be a cheap probe before a long run -- becomes the thing
    that hangs. A trial that times out is recorded as such and the grid moves on.
    """
    sizes = []
    for p in sorted(Path(num_dir).glob("*_num.pickle")):
        sizes.append((p.stat().st_size, int(p.stem.split('_')[0])))
    if not sizes:
        log.error("benchmark: no Step 3 output found; run Step 3 first")
        return []
    sizes.sort()

    # Sample evenly across the size-sorted partitions, so the set spans the
    # small/median/large range rather than clustering.
    #
    # The count matters: with fewer partitions than workers, every worker count
    # above the partition count measures the same thing (one wave, bounded by
    # the slowest partition) and the grid cannot discriminate. Default is
    # tuned to give at least a couple of scheduling waves at typical settings.
    n_pick = max(3, min(n_partitions, len(sizes)))
    idx = np.linspace(0, len(sizes) - 1, n_pick).round().astype(int)
    picks = [sizes[i][1] for i in sorted(set(idx.tolist()))]

    log.info("benchmark: %d partitions (weeks %s)", len(picks), picks)
    log.info("benchmark worker grid: %s  (timeout %ds per trial)",
             worker_grid, timeout)

    results = []
    for nw in worker_grid:
        log.info("  trial: workers=%d ...", nw)
        t0 = time.time()
        try:
            Parallel(n_jobs=nw, backend=backend, timeout=timeout)(
                delayed(to_csv)(i, 'day', dname, ul, uf_dict, list_uf,
                                num_dir, tmp_dir, subsession_mode) for i in picks)
            dt = time.time() - t0
            results.append({"workers": nw, "seconds": round(dt, 2)})
            log.info("    workers=%-3d  %.1fs", nw, dt)
        except Exception as exc:
            log.error("    workers=%-3d  FAILED after %.1fs: %s",
                      nw, time.time() - t0, type(exc).__name__)
            results.append({"workers": nw, "seconds": None,
                            "error": type(exc).__name__})

    ok = [r for r in results if r["seconds"]]
    if ok:
        base = ok[0]["seconds"]
        for r in results:
            r["speedup"] = round(base / r["seconds"], 2) if r["seconds"] else None
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="feature_extraction.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Extract features from the CERT Insider Threat Test Dataset "
            "(r4.1-r6.2).\n"
            "Run from inside a decompressed dataset folder, or pass --data-dir."),
        epilog="""
examples
--------
  # run from inside the dataset folder (release inferred from folder name)
  cd /data/r4.2 && python feature_extraction.py --mode day --workers 8

  # or specify everything explicitly, from anywhere
  python feature_extraction.py \\
      --dataset r4.2 \\
      --data-dir   /data/cert/r4.2 \\
      --output-dir /data/cert/extracted/r4.2 \\
      --mode day --workers 8

  # keep intermediates off the dataset volume (e.g. on a fast local disk)
  python feature_extraction.py \\
      --dataset r5.2 --data-dir /mnt/slow/r5.2 \\
      --output-dir /mnt/fast/extracted/r5.2 \\
      --intermediate-dir /mnt/fast/scratch/r5.2 \\
      --mode day --workers 8

  # resume after an interruption -- same command, completed weeks are skipped
  python feature_extraction.py --data-dir /data/cert/r4.2 --mode day --resume

  # everything the original produced: week, day, session, subsession
  python feature_extraction.py --data-dir /data/cert/r4.2 --mode all --workers 8

  # measure worker scaling before committing to a long run
  python feature_extraction.py --data-dir /data/cert/r6.2 --mode day --benchmark

paths
-----
  --data-dir          input: a decompressed CERT release folder
  --dataset           release name; inferred from the folder name if omitted
  --output-dir        final CSVs        (default: <data-dir>/ExtractedData)
  --intermediate-dir  scratch parent    (default: <data-dir>)
  --checkpoint-dir    resume database   (default: <data-dir>/checkpoints)

  Relative paths are resolved against your current shell directory, not the
  dataset folder. Keep --intermediate-dir and --checkpoint-dir per release;
  they must not be shared between r4.2 and r5.2.

outputs
-------
  <output-dir>/<mode><dataset>.csv            e.g. dayr4.2.csv
  <output-dir>/session{nact,time}<N><ds>.csv  subsessions
  <output-dir>/run_metadata.json

notes
-----
  `insider` is the target label: 0 = normal, >0 = threat scenario number.
  It is never removed by --drop-informational.
  --mode all always retains informational fields, so that `all` represents the
  complete extraction rather than a modelling-specific subset.
""")

    g = p.add_argument_group("dataset and modes")
    g.add_argument("--data-dir", type=Path, default=Path.cwd(),
                   help="CERT dataset folder (default: current directory)")
    g.add_argument("--dataset", choices=VALID_DATASETS, default=None,
                   help="release name; inferred from the folder name if omitted")
    g.add_argument("--mode", nargs="+", default=["all"],
                   choices=ALL_MODES + ["all"],
                   help="temporal representations to extract (default: all)")
    g.add_argument("--subsession-time", type=int, nargs="+",
                   default=DEFAULT_SUBSESSION_TIME,
                   help="subsession durations in minutes (default: 120 240)")
    g.add_argument("--subsession-actions", type=int, nargs="+",
                   default=DEFAULT_SUBSESSION_NACT,
                   help="subsession sizes in actions (default: 25 50)")

    g = p.add_argument_group("execution")
    g.add_argument("--workers", default="8",
                   help="worker processes, or 'auto' (default: 8)")
    g.add_argument("--backend", default="loky",
                   choices=["loky", "threading", "multiprocessing"],
                   help="joblib backend (default: loky)")
    g.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="skip partitions already marked complete (default)")
    g.add_argument("--no-resume", dest="resume", action="store_false",
                   help="recompute everything, ignoring checkpoints")
    g.add_argument("--benchmark", action="store_true",
                   help="time Step 4 across worker counts, then exit")
    g.add_argument("--benchmark-weeks", type=int, default=12, metavar="N",
                   help="partitions per benchmark trial (default: 12). Must "
                        "exceed the largest worker count for the grid to "
                        "discriminate between settings.")

    g = p.add_argument_group("paths")
    g.add_argument("--output-dir", type=Path, default=None,
                   help="final CSVs (default: <data-dir>/ExtractedData)")
    g.add_argument("--intermediate-dir", type=Path, default=None,
                   help="DataByWeek/NumDataByWeek/tmp parent (default: <data-dir>)")
    g.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="checkpoint database (default: <data-dir>/checkpoints)")

    g = p.add_argument_group("output")
    g.add_argument("--drop-informational", action="store_true",
                   help=f"drop {', '.join(INFORMATIONAL_FIELDS)} from the CSVs. "
                        "Ignored for --mode all. Never drops `insider`.")
    g.add_argument("--cleanup", action="store_true",
                   help="delete intermediates after a successful run "
                        "(default: keep, so re-runs are cheap)")

    g = p.add_argument_group("logging")
    g.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    g.add_argument("--log-file", type=Path, default=None)
    g.add_argument("--no-rich", action="store_true",
                   help="plain log-line progress instead of a live bar")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    return p


def setup_logging(level: str, log_file: Path | None, quiet: bool):
    handlers = []
    if not quiet:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level),
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def resolve_modes(requested):
    """
    Expand and order the requested modes.

    'subsession' is produced by the session pass, so requesting it implies the
    session pass runs -- but the session CSV itself is only written if 'session'
    was actually asked for.
    """
    if "all" in requested:
        return ALL_MODES, True
    modes = [m for m in ALL_MODES if m in requested]
    return modes, False


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, args.log_file, args.quiet)

    # Resolve every user-supplied path against the ORIGINAL working directory,
    # before the chdir below. Otherwise a relative --output-dir would silently
    # resolve against the dataset folder instead of where the user typed it.
    data_dir = args.data_dir.resolve()
    user_out = args.output_dir.resolve() if args.output_dir else None
    user_inter = args.intermediate_dir.resolve() if args.intermediate_dir else None
    user_ckpt = args.checkpoint_dir.resolve() if args.checkpoint_dir else None

    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir not found: {data_dir}")

    dname = args.dataset or data_dir.name
    if dname not in VALID_DATASETS:
        raise SystemExit(
            f"Cannot determine the CERT release.\n"
            f"  --data-dir is '{data_dir}' (folder name: '{data_dir.name}')\n"
            f"  Either run from / point at a folder named after the release "
            f"(e.g. r4.2), or pass --dataset explicitly.\n"
            f"  Valid: {', '.join(VALID_DATASETS)}")

    # The original resolves LDAP/, *.csv and answers/ relatively, so the process
    # must run with the dataset folder as its working directory.
    os.chdir(data_dir)

    inter = user_inter or data_dir
    week_dir = inter / "DataByWeek"
    num_dir = inter / "NumDataByWeek"
    tmp_dir = inter / "tmp"
    out_dir = user_out or (data_dir / "ExtractedData")
    ckpt_dir = user_ckpt or (data_dir / "checkpoints")
    for d in (week_dir, num_dir, tmp_dir, out_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    workers = (os.cpu_count() or 8) if args.workers == "auto" else int(args.workers)
    modes, is_all = resolve_modes(args.mode)

    subsession_mode = {}
    if "subsession" in modes:
        if args.subsession_actions:
            subsession_mode['nact'] = list(args.subsession_actions)
        if args.subsession_time:
            subsession_mode['time'] = list(args.subsession_time)

    # --mode all always keeps informational fields: `all` is defined as the
    # complete extraction, not a modelling-ready subset.
    drop_info = args.drop_informational and not is_all

    n_weeks = NUM_WEEKS[dname]
    ckpt = CheckpointDB(ckpt_dir / "checkpoints.sqlite", dname)
    run_started = datetime.now()
    t_run = time.time()

    log.info("CERT feature extraction  v%s", __version__)
    log.info("  dataset        : %s  (%d weeks)", dname, n_weeks)
    log.info("  input          : %s", data_dir)
    log.info("  output         : %s", out_dir)
    log.info("  intermediates  : %s", inter)
    log.info("  checkpoints    : %s", ckpt_dir)
    log.info("  modes          : %s%s", ", ".join(modes), "   [all]" if is_all else "")
    log.info("  workers        : %d (%s)", workers, args.backend)
    log.info("  resume         : %s", "on" if args.resume else "off")
    log.info("  informational  : %s", "dropped" if drop_info else "retained")
    if is_all and args.drop_informational:
        log.info("  note: --drop-informational is disabled for --mode all")
    if subsession_mode:
        log.info("  subsessions    : %s", subsession_mode)

    stage_times = {}
    try:
        with ProgressReporter(enabled=not args.no_rich) as progress:

            # ── Step 1 ────────────────────────────────────────────────────────
            t0 = time.time()
            step1_split_by_week(dname, week_dir, ckpt, progress, args.resume)
            stage_times["step1_split_by_week"] = round(time.time() - t0, 1)
            log.info("Step 1 done in %.1f min", stage_times["step1_split_by_week"] / 60)

            # ── Step 2 ────────────────────────────────────────────────────────
            t0 = time.time()
            users = get_mal_userdata(dname, work_dir=data_dir, week_dir=str(week_dir))
            stage_times["step2_user_list"] = round(time.time() - t0, 1)
            log.info("Step 2 done in %.1f min -- %d users",
                     stage_times["step2_user_list"] / 60, len(users))

            # ── Step 3 ────────────────────────────────────────────────────────
            t0 = time.time()
            done3 = ckpt.completed("step3", "-") if args.resume else set()
            todo3 = [i for i in range(n_weeks)
                     if not (i in done3 and (num_dir / f"{i}_num.pickle").is_file())]
            log.info("Step 3: %d/%d partitions pending (%d reused)",
                     len(todo3), n_weeks, n_weeks - len(todo3))

            if todo3:
                progress.add("step3", "Step 3  numeric conversion", len(todo3))

                def _run3(i):
                    s = time.time()
                    try:
                        rows = process_week_num(i, users, num_dir, data=dname,
                                                week_dir=str(week_dir))
                        return (i, "completed", time.time() - s, rows, None)
                    except Exception as exc:                # isolate the partition
                        return (i, "failed", time.time() - s, None, repr(exc))

                for res in Parallel(n_jobs=workers, backend=args.backend,
                                    return_as="generator")(
                        delayed(_run3)(i) for i in todo3):
                    i, status, dur, rows, err = res
                    ckpt.mark("step3", "-", i, status, duration=dur, rows=rows,
                              output_path=str(num_dir / f"{i}_num.pickle"),
                              error=err)
                    if status == "failed":
                        log.error("Step 3 week %d FAILED: %s", i, err)
                    progress.advance("step3", 1,
                                     rate_hint=f"{rows:,} rows" if rows else "failed")

            log.info("Step 3: all partitions consumed, tearing down workers")

            s3 = ckpt.summary("step3", "-")
            log.info("Step 3 summary: %s", s3)
            stage_times["step3_numeric"] = round(time.time() - t0, 1)
            log.info("Step 3 done in %.1f min", stage_times["step3_numeric"] / 60)

            # ── Benchmark short-circuit ───────────────────────────────────────
            log.info("building user feature dictionaries")
            (ul, uf_dict, list_uf) = get_u_features_dicts(users, data=dname)
            log.info("user feature dictionaries ready")
            if args.benchmark:
                # Never exceed the logical CPU count. Spawning 2x cores of
                # interpreters to run three tasks measures process startup, not
                # throughput, and is where this previously hung.
                ncpu = os.cpu_count() or workers
                grid = sorted({g for g in (4, 8, workers, ncpu) if 1 <= g <= ncpu})
                res = run_benchmark(dname, users, ul, uf_dict, list_uf,
                                    num_dir, tmp_dir, subsession_mode, grid,
                                    backend=args.backend,
                                    n_partitions=args.benchmark_weeks)
                out = out_dir / f"benchmark_{dname}.json"
                out.write_text(json.dumps(res, indent=2))
                log.info("benchmark written to %s", out)
                print(json.dumps(res, indent=2))
                return 0

            # ── Step 4 ────────────────────────────────────────────────────────
            # 'subsession' is not its own pass; it is produced by the session
            # pass. Run session once if either was requested.
            passes = [m for m in modes if m in ("week", "day", "session")]
            if "subsession" in modes and "session" not in passes:
                passes.append("session")

            for mode in passes:
                t0 = time.time()
                # Week 0 has no complete preceding week, so week-mode starts at 1.
                week_range = list(range(0, n_weeks)) if mode in ("day", "session") \
                    else list(range(1, n_weeks))

                donem = ckpt.completed("step4", mode) if args.resume else set()
                todo = [i for i in week_range
                        if not (i in donem and (tmp_dir / f"{i}{mode}.pickle").is_file())]
                log.info("Step 4 [%s]: %d/%d partitions pending (%d reused)",
                         mode, len(todo), len(week_range), len(week_range) - len(todo))

                if todo:
                    progress.add(f"step4-{mode}", f"Step 4  {mode:<8}", len(todo))
                    ss = subsession_mode if mode == "session" else {}

                    def _run4(i, _mode=mode, _ss=ss):
                        s = time.time()
                        try:
                            rows = to_csv(i, _mode, dname, ul, uf_dict, list_uf,
                                          num_dir, tmp_dir, _ss)
                            return (i, "completed", time.time() - s, rows, None)
                        except Exception as exc:
                            return (i, "failed", time.time() - s, None, repr(exc))

                    for res in Parallel(n_jobs=workers, backend=args.backend,
                                        return_as="generator")(
                            delayed(_run4)(i) for i in todo):
                        i, status, dur, rows, err = res
                        ckpt.mark("step4", mode, i, status, duration=dur, rows=rows,
                                  output_path=str(tmp_dir / f"{i}{mode}.pickle"),
                                  error=err)
                        if status == "failed":
                            log.error("Step 4 [%s] week %d FAILED: %s", mode, i, err)
                        progress.advance(f"step4-{mode}", 1,
                                         rate_hint=f"{rows:,} rows" if rows else "failed")

                # Consolidate. Only write the CSV for modes actually requested:
                # session may have run purely to produce subsessions.
                if mode in modes:
                    out = consolidate(week_range, mode, dname, tmp_dir, out_dir,
                                      drop_informational=drop_info)
                    log.info("  wrote %s", out)

                if mode == "session" and "subsession" in modes and subsession_mode:
                    for k1 in subsession_mode:
                        for k2 in subsession_mode[k1]:
                            out = consolidate(week_range, mode, dname, tmp_dir,
                                              out_dir, suffix=f"{k1}{k2}",
                                              drop_informational=drop_info)
                            log.info("  wrote %s", out)

                stage_times[f"step4_{mode}"] = round(time.time() - t0, 1)
                log.info("Step 4 [%s] done in %.1f min",
                         mode, stage_times[f"step4_{mode}"] / 60)

            # Slowest partitions are the ones worth profiling next.
            slow = ckpt.durations("step4", passes[-1])[:3] if passes else []
            if slow:
                log.info("slowest %s partitions: %s", passes[-1],
                         ", ".join(f"week {w} ({d:.0f}s)" for w, d in slow))

    finally:
        # ── Run metadata ──────────────────────────────────────────────────────
        meta = {
            "script_version": __version__,
            "dataset": dname,
            "n_weeks": n_weeks,
            "modes_requested": args.mode,
            "modes_run": modes,
            "subsession_mode": subsession_mode,
            "informational_fields": "dropped" if drop_info else "retained",
            "workers": workers,
            "backend": args.backend,
            "resume": args.resume,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "libraries": {"pandas": pd.__version__,
                          "numpy": np.__version__},
            "started_at": run_started.isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "total_seconds": round(time.time() - t_run, 1),
            "stage_seconds": stage_times,
            "checkpoint_db": str(ckpt_dir / "checkpoints.sqlite"),
            "output_dir": str(out_dir),
        }
        (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
        ckpt.close()

    if args.cleanup:
        log.warning("--cleanup: deleting %s, %s, %s", week_dir, num_dir, tmp_dir)
        for d in (week_dir, num_dir, tmp_dir):
            shutil.rmtree(d, ignore_errors=True)
    else:
        log.info("intermediates retained (use --cleanup to remove)")

    log.info("done in %.1f min", (time.time() - t_run) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())