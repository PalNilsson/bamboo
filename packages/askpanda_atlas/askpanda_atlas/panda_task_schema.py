"""PanDA task JSON schema constants and object model.

This module is intentionally free of any bamboo-core dependency so it can
be imported by plugin implementations, fallback stubs, and unit tests alike.

The three public classes — ``PandaJob``, ``PandaTaskData``, and the
extraction helpers — provide a stable narrow interface over the raw JSON
returned by ``GET /jobs/?jeditaskid={id}&json``.  Unknown fields are
preserved in ``.extra`` for forward compatibility.

Typical usage::

    data = fetch_jsonish(url)          # raw dict from HTTP
    task = PandaTaskData(data)
    ids  = get_pandaid_list(task)      # [7061545370, ...]
    evid = build_evidence(task)        # compact dict safe to send to an LLM
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Schema dictionaries — used both as documentation and as LLM context aids.
# ---------------------------------------------------------------------------

TOP_LEVEL_SCHEMA: dict[str, str] = {
    "selectionsummary": "Aggregated summary of field values across the task",
    "jobs": "List of job records",
    "errsByCount": "Error summaries grouped by count",
}

SELECTIONSUMMARY_ITEM_SCHEMA: dict[str, str] = {
    "field": "Field name being summarized",
    "list": "List of value/count entries for the field",
    "stats": "Optional aggregate statistics for the field",
}

JOB_FIELD_SCHEMA: dict[str, str] = {
    "actualcorecount": "Actual number of CPU cores used",
    "assignedpriority": "Priority assigned to the job",
    "atlasrelease": "ATLAS release used by the job",
    "attemptnr": "Attempt number",
    "avgpss": "Average proportional set size",
    "avgrss": "Average resident set size",
    "avgswap": "Average swap usage",
    "avgvmem": "Average virtual memory usage",
    "avgvmemmb": "Average virtual memory usage in MB",
    "batchid": "Batch system identifier",
    "brokerageerrorcode": "Brokerage error code",
    "brokerageerrordiag": "Brokerage error diagnostics",
    "category": "Job category",
    "cloud": "Cloud/region assignment",
    "cmtconfig": "CMT configuration string",
    "commandtopilot": "Command sent to the pilot",
    "computingelement": "Computing element name",
    "computingsite": "Computing site name",
    "consumer": "Consumer identifier or label",
    "container_name": "Container image or runtime container name",
    "corecount": "Requested core count",
    "countrygroup": "Country group",
    "cpu_architecture_level": "CPU architecture feature level",
    "cpuconsumptiontime": "CPU consumption time",
    "cpuconsumptionunit": "CPU consumption unit string",
    "cpuconversion": "CPU conversion factor",
    "cpuefficiency": "CPU efficiency percentage",
    "creationhost": "Host where the job record was created",
    "creationtime": "Job creation time",
    "currentpriority": "Current job priority",
    "ddmerrorcode": "DDM error code",
    "ddmerrordiag": "DDM error diagnostics",
    "destinationdblock": "Destination data block",
    "destinationse": "Destination storage element",
    "destinationsite": "Destination site",
    "diskio": "Disk I/O metric",
    "dispatchdblock": "Dispatch data block",
    "duration": "Job duration as a formatted string",
    "durationmin": "Job duration in minutes",
    "durationsec": "Job duration in seconds",
    "endtime": "Job end time",
    "eventservice": "Event service mode",
    "exeerrorcode": "Payload execution error code",
    "exeerrordiag": "Payload execution diagnostics",
    "failedattempt": "Failed attempt count or flag",
    "gco2_global": "Estimated global CO2 emission",
    "gco2_regional": "Estimated regional CO2 emission",
    "grid": "Grid label",
    "gshare": "Global share or fair-share group",
    "homecloud": "Home cloud assignment",
    "homepackage": "Home software package",
    "hs06": "HS06 benchmark value",
    "hs06sec": "HS06-seconds consumed",
    "inputfilebytes": "Input file bytes",
    "inputfileproject": "Input file project",
    "inputfiletype": "Input file type",
    "ipconnectivity": "IP connectivity status",
    "jeditaskid": "JEDI task ID",
    "job_label": "Job label",
    "jobdefinitionid": "Job definition ID",
    "jobdispatchererrorcode": "Job dispatcher error code",
    "jobdispatchererrordiag": "Job dispatcher error diagnostics",
    "jobexecutionid": "Job execution ID",
    "jobinfo": "Free-form job info string",
    "jobmetrics": "Job metrics string",
    "jobname": "Job name",
    "jobsetid": "Job set ID",
    "jobsetrange": "Job set range",
    "jobstatus": "Job status",
    "jobsubstatus": "Job sub-status",
    "lockedby": "Lock owner or system",
    "maxattempt": "Maximum number of attempts",
    "maxcpucount": "Maximum CPU count",
    "maxcpuunit": "Maximum CPU unit",
    "maxdiskcount": "Maximum disk count",
    "maxdiskunit": "Maximum disk unit",
    "maxpss": "Maximum proportional set size",
    "maxpssgbpercore": "Maximum PSS per core in GB",
    "maxrss": "Maximum resident set size",
    "maxswap": "Maximum swap usage",
    "maxvmem": "Maximum virtual memory usage",
    "maxvmemmb": "Maximum virtual memory usage in MB",
    "maxwalltime": "Maximum wall time",
    "meancorecount": "Mean core count used",
    "memoryleak": "Memory leak indicator or value",
    "memoryleakx2": "Secondary memory leak indicator or value",
    "minramcount": "Minimum RAM count",
    "minramunit": "Minimum RAM unit",
    "modificationhost": "Host where the job record was modified",
    "modificationtime": "Job modification time",
    "nevents": "Number of events processed",
    "ninputdatafiles": "Number of input data files",
    "ninputfiles": "Number of input files",
    "noutputdatafiles": "Number of output data files",
    "nucleus": "Nucleus assignment",
    "outputfilebytes": "Output file bytes",
    "outputfiletype": "Output file type",
    "pandaid": "PanDA job ID",
    "parentid": "Parent job ID",
    "piloterrorcode": "Pilot error code",
    "piloterrordiag": "Pilot error diagnostics",
    "pilotid": "Pilot ID or pilot log reference",
    "pilottiming": "Pilot timing breakdown",
    "pilotversion": "Pilot version",
    "priorityrange": "Priority range",
    "processingtype": "Processing type",
    "processor_type": "Processor type",
    "proddblock": "Production data block",
    "proddbupdatetime": "Production DB update time",
    "prodserieslabel": "Production series label",
    "prodsourcelabel": "Production source label",
    "produserid": "Production user DN or user ID",
    "produsername": "Production user name",
    "raterbytes": "Read rate in bytes",
    "raterchar": "Read rate in characters",
    "ratewbytes": "Write rate in bytes",
    "ratewchar": "Write rate in characters",
    "relocationflag": "Relocation flag",
    "reqid": "Request ID",
    "resourcetype": "Resource type",
    "schedulerid": "Scheduler ID",
    "sourcesite": "Source site",
    "specialhandling": "Special handling flags",
    "starttime": "Job start time",
    "statechangetime": "Last state change time",
    "superrorcode": "Super error code",
    "superrordiag": "Super error diagnostics",
    "taskbuffererrorcode": "Task buffer error code",
    "taskbuffererrordiag": "Task buffer error diagnostics",
    "taskid": "Task-specific job index",
    "totrbytes": "Total bytes read",
    "totrchar": "Total characters read",
    "totwbytes": "Total bytes written",
    "totwchar": "Total characters written",
    "transexitcode": "Transformation exit code",
    "transfertype": "Transfer type",
    "transformation": "Transformation name",
    "vo": "Virtual organization",
    "waittime": "Wait time as a formatted string",
    "waittimesec": "Wait time in seconds",
    "wn": "Worker node hostname",
    "workinggroup": "Working group",
    "workqueue_id": "Work queue ID",
}

SCHEMA: dict[str, dict[str, str]] = {
    "top_level": TOP_LEVEL_SCHEMA,
    "selectionsummary_item": SELECTIONSUMMARY_ITEM_SCHEMA,
    "job_fields": JOB_FIELD_SCHEMA,
}

# Convenience tuples for membership tests.
EXPECTED_TOP_LEVEL_KEYS: tuple[str, ...] = tuple(TOP_LEVEL_SCHEMA)
EXPECTED_SELECTIONSUMMARY_ITEM_KEYS: tuple[str, ...] = tuple(SELECTIONSUMMARY_ITEM_SCHEMA)
EXPECTED_JOB_FIELDS: tuple[str, ...] = tuple(JOB_FIELD_SCHEMA)

# ---------------------------------------------------------------------------
# Slim per-job field set — the subset written into evidence dicts.
# Keeping this narrow prevents the LLM context from exploding on large tasks.
# ---------------------------------------------------------------------------

_SLIM_JOB_FIELDS: tuple[str, ...] = (
    "pandaid",
    "jobstatus",
    "jobsubstatus",
    "computingsite",
    "attemptnr",
    "piloterrorcode",
    "piloterrordiag",
    "exeerrorcode",
    "exeerrordiag",
    "jobdispatchererrorcode",
    "jobdispatchererrordiag",
    "taskbuffererrorcode",
    "taskbuffererrordiag",
    "brokerageerrorcode",
    "brokerageerrordiag",
    "starttime",
    "endtime",
    "durationsec",
    "jobname",
    "transformation",
    "cloud",
)

# How many failed / finished jobs to include verbatim in evidence.
_MAX_FAILED_JOBS: int = 20
_MAX_FINISHED_JOBS: int = 5
_MAX_PANDAID_LIST_INLINE: int = 500  # beyond this, the list is noted but not inlined


# ---------------------------------------------------------------------------
# Object model
# ---------------------------------------------------------------------------


class PandaJob:
    """Represents a single PanDA job record from the task jobs endpoint.

    The class promotes the most-queried fields to typed attributes while
    preserving the full raw payload in ``_raw`` and unknown fields in
    ``extra``.

    Attributes:
        pandaid: PanDA job identifier.
        jobstatus: Current job status string (e.g. ``"failed"``, ``"finished"``).
        jeditaskid: JEDI task identifier owning this job.
        reqid: Request identifier.
        processingtype: Processing type string.
        attemptnr: Attempt number for this job within its task.
        computingsite: Name of the computing site where the job ran.
        piloterrorcode: Pilot-level error code (0 = no error).
        piloterrordiag: Human-readable pilot error description.
        exeerrorcode: Payload execution error code.
        exeerrordiag: Human-readable payload execution error description.
        extra: All fields not explicitly promoted, keyed by field name.
    """

    # Fields promoted to typed attributes.
    _PROMOTED: frozenset[str] = frozenset({
        "pandaid",
        "jobstatus",
        "jeditaskid",
        "reqid",
        "processingtype",
        "attemptnr",
        "computingsite",
        "piloterrorcode",
        "piloterrordiag",
        "exeerrorcode",
        "exeerrordiag",
    })

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialise a PandaJob from a raw job dictionary.

        Args:
            data: Raw job dictionary as returned by the BigPanDA API.
        """
        self.pandaid: int | None = _safe_int(data.get("pandaid"))
        self.jobstatus: str | None = data.get("jobstatus")
        self.jeditaskid: int | None = _safe_int(data.get("jeditaskid"))
        self.reqid: int | None = _safe_int(data.get("reqid"))
        self.processingtype: str | None = data.get("processingtype")
        self.attemptnr: int | None = _safe_int(data.get("attemptnr"))
        self.computingsite: str | None = data.get("computingsite")
        self.piloterrorcode: int | None = _safe_int(data.get("piloterrorcode"))
        self.piloterrordiag: str | None = data.get("piloterrordiag")
        self.exeerrorcode: int | None = _safe_int(data.get("exeerrorcode"))
        self.exeerrordiag: str | None = data.get("exeerrordiag")

        self._raw: dict[str, Any] = data
        self.extra: dict[str, Any] = {
            k: v for k, v in data.items() if k not in self._PROMOTED
        }

    @classmethod
    def schema(cls) -> dict[str, str]:
        """Return a copy of the complete job-field schema dictionary.

        Returns:
            Mapping of field name → concise human description.
        """
        return dict(JOB_FIELD_SCHEMA)

    def get(self, key: str, default: Any = None) -> Any:
        """Return any field from the original job payload by name.

        Args:
            key: Field name to look up.
            default: Value returned when the key is absent.

        Returns:
            Field value or *default*.
        """
        return self._raw.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the original, unmodified job dictionary.

        Returns:
            Shallow copy of the raw job payload.
        """
        return dict(self._raw)

    def to_slim_dict(self) -> dict[str, Any]:
        """Return a reduced dictionary containing only evidence-relevant fields.

        Only the fields listed in ``_SLIM_JOB_FIELDS`` are included.
        ``None``-valued fields are omitted to keep the output compact.

        Returns:
            Dict of non-None evidence fields.
        """
        return {
            k: self._raw[k]
            for k in _SLIM_JOB_FIELDS
            if k in self._raw and self._raw[k] is not None
        }

    def __repr__(self) -> str:  # pragma: no cover
        """Return a concise developer-readable representation of this job."""
        return (
            f"PandaJob(pandaid={self.pandaid}, "
            f"status={self.jobstatus}, "
            f"site={self.computingsite}, "
            f"taskid={self.jeditaskid})"
        )


class PandaTaskData:
    """Represents the full response from ``GET /jobs/?jeditaskid={id}&json``.

    The three top-level keys — ``jobs``, ``selectionsummary``,
    ``errsByCount`` — are parsed into typed collections.  Anything else is
    preserved in ``extra``.

    Attributes:
        jobs: Ordered list of :class:`PandaJob` objects.
        selectionsummary: Raw list of per-field aggregation dicts.
        errs_by_count: Raw error-summary mapping keyed by error code.
        extra: Any top-level keys not in the known schema.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialise from the raw API response dictionary.

        Args:
            data: Parsed JSON dict from the BigPanDA jobs endpoint.
        """
        self.jobs: list[PandaJob] = [
            PandaJob(j) for j in data.get("jobs", [])
        ]
        self.selectionsummary: list[dict[str, Any]] = data.get("selectionsummary", [])
        self.errs_by_count: Any = data.get("errsByCount", {})
        self._raw: dict[str, Any] = data
        self.extra: dict[str, Any] = {
            k: v
            for k, v in data.items()
            if k not in {"jobs", "selectionsummary", "errsByCount"}
        }

    @classmethod
    def schema(cls) -> dict[str, dict[str, str]]:
        """Return a copy of the full nested schema dictionary.

        Returns:
            Dict with keys ``"top_level"``, ``"selectionsummary_item"``,
            and ``"job_fields"``, each mapping field names to descriptions.
        """
        return {
            "top_level": dict(TOP_LEVEL_SCHEMA),
            "selectionsummary_item": dict(SELECTIONSUMMARY_ITEM_SCHEMA),
            "job_fields": dict(JOB_FIELD_SCHEMA),
        }

    def get_job(self, pandaid: int) -> PandaJob | None:
        """Return the :class:`PandaJob` with the given PanDA ID, or ``None``.

        Args:
            pandaid: The PanDA job identifier to search for.

        Returns:
            Matching :class:`PandaJob`, or ``None`` if not found.
        """
        return next((j for j in self.jobs if j.pandaid == pandaid), None)

    def to_dict(self) -> dict[str, Any]:
        """Return the original, unmodified task dictionary.

        Returns:
            Shallow copy of the raw API response dict.
        """
        return dict(self._raw)

    def __repr__(self) -> str:  # pragma: no cover
        """Return a concise developer-readable representation of this task."""
        first = self.jobs[0].pandaid if self.jobs else None
        return (
            f"PandaTaskData(total_jobs={len(self.jobs)}, "
            f"first_pandaid={first})"
        )


# ---------------------------------------------------------------------------
# Extraction helpers — pure functions, no I/O
# ---------------------------------------------------------------------------


def _safe_int(value: Any) -> int | None:
    """Convert *value* to ``int``, returning ``None`` on failure.

    Args:
        value: Any value to attempt conversion on.

    Returns:
        Integer form of *value*, or ``None``.
    """
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def get_pandaid_list(task: PandaTaskData) -> list[int]:
    """Return every PanDA job ID present in the task, in order.

    The list can be very large (thousands of entries) for tasks with many
    jobs.  Callers that only need IDs for a particular status should use
    :func:`get_pandaid_list_by_status` instead.

    Args:
        task: A fully parsed :class:`PandaTaskData` instance.

    Returns:
        Ordered list of integer PanDA job IDs.  IDs that could not be
        parsed as integers are silently omitted.
    """
    return [j.pandaid for j in task.jobs if j.pandaid is not None]


def get_pandaid_list_by_status(
    task: PandaTaskData,
    status: str,
) -> list[int]:
    """Return PanDA job IDs filtered to a single job status.

    Args:
        task: A fully parsed :class:`PandaTaskData` instance.
        status: Status string to match (case-sensitive, e.g. ``"failed"``).

    Returns:
        Ordered list of integer PanDA job IDs whose ``jobstatus`` equals
        *status*.  IDs that could not be parsed as integers are omitted.
    """
    return [
        j.pandaid
        for j in task.jobs
        if j.jobstatus == status and j.pandaid is not None
    ]


def summarise_selectionsummary(
    selectionsummary: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flatten the ``selectionsummary`` array into a plain key→value dict.

    The ``selectionsummary`` from the API is a list of objects like::

        {"field": "taskname", "list": [{"value": "...", "count": N}], ...}

    For single-value fields (count == 1 unique value) the value is stored
    directly.  For multi-value fields the full list is preserved so callers
    can see the distribution.

    Args:
        selectionsummary: The raw ``selectionsummary`` list from the API.

    Returns:
        Flat dict mapping field name to its dominant value or value list.
    """
    result: dict[str, Any] = {}
    for item in selectionsummary:
        field = item.get("field")
        if not field:
            continue
        entries: list[dict[str, Any]] = item.get("list", [])
        if len(entries) == 1:
            result[field] = entries[0].get("value")
        elif entries:
            result[field] = [e.get("value") for e in entries]
    return result


def _count_jobs_by(
    jobs: list[PandaJob],
    attr: str,
) -> dict[str, int]:
    """Count jobs grouped by a string attribute.

    Args:
        jobs: List of :class:`PandaJob` instances to aggregate.
        attr: Attribute name to group by (e.g. ``"jobstatus"``).

    Returns:
        Dict mapping attribute value → job count, sorted by count descending.
    """
    counter: Counter[str] = Counter()
    for job in jobs:
        val = getattr(job, attr, None) or job.get(attr)
        if val is not None:
            counter[str(val)] += 1
    return dict(counter.most_common())


def _sample_jobs(
    jobs: list[PandaJob],
    status: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return up to *limit* slim dicts for jobs matching *status*.

    Args:
        jobs: Full list of :class:`PandaJob` instances.
        status: Job status to filter on.
        limit: Maximum number of jobs to include.

    Returns:
        List of slim job dicts (see :meth:`PandaJob.to_slim_dict`).
    """
    return [
        j.to_slim_dict()
        for j in jobs
        if j.jobstatus == status
    ][:limit]


def build_evidence(task: PandaTaskData) -> dict[str, Any]:
    """Build a compact evidence dictionary safe to embed in an LLM prompt.

    The goal is maximum signal at minimum token cost.  The structure is::

        {
            "task_summary": { ... },          # from selectionsummary
            "total_jobs": N,
            "jobs_by_status": { ... },        # counts
            "jobs_by_site": { ... },          # counts
            "jobs_by_piloterrorcode": { ... },# counts
            "errs_by_count": { ... },         # raw from API
            "failed_jobs_sample": [ ... ],    # up to 20, slim dicts
            "finished_jobs_sample": [ ... ],  # up to 5, slim dicts
            "pandaid_list_note": "...",        # note if list too large to inline
            "pandaid_list": [ ... ] | None,   # inlined only when ≤ 500 jobs
        }

    Args:
        task: A fully parsed :class:`PandaTaskData` instance.

    Returns:
        Dict ready to be JSON-serialised and passed as tool evidence.
    """
    jobs = task.jobs
    total = len(jobs)

    task_summary = summarise_selectionsummary(task.selectionsummary)

    jobs_by_status = _count_jobs_by(jobs, "jobstatus")
    jobs_by_site = _count_jobs_by(jobs, "computingsite")

    # piloterrorcode is an int attribute; build its counter manually.
    pilot_err_counter: Counter[str] = Counter()
    for job in jobs:
        code = job.piloterrorcode
        if code is not None and code != 0:
            pilot_err_counter[str(code)] += 1
    jobs_by_piloterrorcode: dict[str, int] = dict(pilot_err_counter.most_common())

    failed_sample = _sample_jobs(jobs, "failed", _MAX_FAILED_JOBS)
    finished_sample = _sample_jobs(jobs, "finished", _MAX_FINISHED_JOBS)

    # Inline the pandaid list only when it won't overwhelm the context.
    all_ids = get_pandaid_list(task)
    if total <= _MAX_PANDAID_LIST_INLINE:
        pandaid_list: list[int] | None = all_ids
        pandaid_note = f"All {total} PanDA job IDs inlined."
    else:
        pandaid_list = None
        pandaid_note = (
            f"Task has {total} jobs; PanDA ID list not inlined to save tokens. "
            f"Use get_pandaid_list() or get_pandaid_list_by_status() to retrieve IDs."
        )

    return {
        "task_summary": task_summary,
        "total_jobs": total,
        "jobs_by_status": jobs_by_status,
        "jobs_by_site": jobs_by_site,
        "jobs_by_piloterrorcode": jobs_by_piloterrorcode,
        "errs_by_count": task.errs_by_count,
        "failed_jobs_sample": failed_sample,
        "finished_jobs_sample": finished_sample,
        "pandaid_list_note": pandaid_note,
        "pandaid_list": pandaid_list,
    }
