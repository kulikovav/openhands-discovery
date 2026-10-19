import asyncio

import httpx

from openhands_discovery.runs import (
    RUN_PAGE_SIZE,
    AutomationRunsPage,
    ChatRuns,
    NewestRuns,
    SingleRun,
    apply_conversation_records,
    conversation_runs,
    fetch_conversation_records,
    fetch_run_conversation_records,
    merge_runs,
    newest_automations,
    read_run_pages,
    run_by_id,
    run_conversation_ids,
    select_scan,
)


def test_merge_runs_sorts_filters_and_sums_counters():
    pages = [
        (
            {"id": "a1", "name": "nightly"},
            AutomationRunsPage(
                runs=[{"id": "r1", "created_at": "2026-10-01T00:00:00Z", "status": "RUNNING"}],
                status_counts={"RUNNING": 1},
            ),
        ),
        (
            {"id": "a2", "name": "hourly"},
            AutomationRunsPage(
                runs=[
                    {"id": "r2", "created_at": "2026-10-02T00:00:00Z", "status": "COMPLETED"},
                    {"id": "r3", "created_at": "2026-09-30T00:00:00Z", "status": "FAILED"},
                    {"id": 7, "created_at": "2026-10-03T00:00:00Z", "status": "FAILED"},
                    {"created_at": "2026-10-04T00:00:00Z", "status": "FAILED"},
                ],
                status_counts={"COMPLETED": 1, "FAILED": "two"},
            ),
        ),
    ]

    runs, status_counts = merge_runs(pages, limit=2)
    assert [row["id"] for row in runs] == ["7", "r2"]
    assert runs[0]["automation_name"] == "hourly"
    assert status_counts == {"RUNNING": 1, "COMPLETED": 1}

    filtered, _ = merge_runs(pages, limit=1, status="FAILED")
    assert [row["id"] for row in filtered] == ["7"]


def test_select_scan_prefers_the_run_lookup():
    assert isinstance(select_scan(5, False, None), NewestRuns)
    assert select_scan(5, False, None).page_size == 5
    assert select_scan(500, False, None).page_size == RUN_PAGE_SIZE
    assert isinstance(select_scan(5, True, None), ChatRuns)
    assert isinstance(select_scan(5, True, "r1"), SingleRun)
    assert select_scan(5, True, "r1").run_id == "r1"


def test_chat_runs_follows_pages_until_the_limit():
    offsets = []

    def handler(request):
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        if offset == 0:
            return httpx.Response(
                200,
                json={
                    "runs": [{"id": "r1"}, {"id": "r2", "conversation_id": "c2"}],
                    "total": 4,
                    "status_counts": {"RUNNING": 1},
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "runs": [{"id": "r3", "conversation_id": "c3"}],
                "total": 4,
                "status_counts": {"RUNNING": 2},
            },
            request=request,
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ChatRuns(2).read_page(client, "http://127.0.0.1:8000", "key", "a1")

    page = asyncio.run(run())

    assert offsets == [0, 2]
    assert [row["id"] for row in page.runs] == ["r2", "r3"]
    assert page.status_counts == {"RUNNING": 1}


def test_chat_runs_stops_at_the_automation_total():
    offsets = []

    def handler(request):
        offsets.append(int(request.url.params["offset"]))
        return httpx.Response(
            200,
            json={"runs": [{"id": "r1", "conversation_id": "c1"}], "total": 1},
            request=request,
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ChatRuns(5).read_page(client, "http://127.0.0.1:8000", "key", "a1")

    page = asyncio.run(run())

    assert offsets == [0]
    assert [row["id"] for row in page.runs] == ["r1"]


def test_single_run_follows_pages_until_found():
    offsets = []

    def handler(request):
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        if offset == 0:
            return httpx.Response(
                200,
                json={"runs": [{"id": "r1"}, {"id": "r2"}], "total": 4},
                request=request,
            )
        return httpx.Response(
            200,
            json={"runs": [{"id": "r3"}, {"id": "r4"}], "total": 4},
            request=request,
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await SingleRun("r3").read_page(client, "http://127.0.0.1:8000", "key", "a1")

    page = asyncio.run(run())

    assert offsets == [0, 2]
    assert [row["id"] for row in page.runs] == ["r3"]


def test_single_run_returns_empty_when_the_automation_ends():
    def handler(request):
        return httpx.Response(200, json={"runs": [], "total": 0}, request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await SingleRun("missing").read_page(client, "http://127.0.0.1:8000", "key", "a1")

    assert asyncio.run(run()).runs == []


def test_read_run_pages_reports_one_failed_automation():
    def handler(request):
        if "/bad/runs" in request.url.path:
            return httpx.Response(500, request=request)
        return httpx.Response(
            200,
            json={"runs": [{"id": "r1", "created_at": "2026-10-01T00:00:00Z"}]},
            request=request,
        )

    automations = [{"id": "a1", "name": "one"}, {"id": "bad", "name": "broken"}, {"id": "a2", "name": "two"}]

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await read_run_pages(client, "http://127.0.0.1:8000", "key", automations, NewestRuns(100))

    result = asyncio.run(run())

    assert [automation["id"] for automation, _ in result.pages] == ["a1", "a2"]
    assert [(entry.id, entry.name) for entry in result.unavailable] == [("bad", "broken")]


def test_newest_automations_caps_the_scan():
    ordered = newest_automations(
        [
            {"id": "a1", "created_at": "2026-10-01T00:00:00Z"},
            {"id": "a2", "created_at": "2026-10-02T00:00:00Z"},
        ],
        cap=1,
    )
    assert [automation["id"] for automation in ordered] == ["a2"]


def test_run_page_helpers_keep_only_rows_with_the_selected_shape():
    rows = [
        {"id": "r1"},
        {"id": "r2", "conversation_id": "c2"},
        {"id": "r3", "conversation_id": ""},
        {"id": "r4", "conversation_id": "c4"},
    ]

    assert [row["id"] for row in conversation_runs(rows)] == ["r2", "r4"]
    assert conversation_runs([]) == []
    assert run_by_id(rows, "r2") == [rows[1]]
    assert run_by_id(rows, "missing") == []
    assert run_conversation_ids(rows) == ["c2", "c4"]
    assert run_conversation_ids([{"conversation_id": "c2"}, {"conversation_id": "c2"}]) == ["c2"]


def test_apply_conversation_records_leaves_unanswered_rows_unset():
    runs = [
        {"id": "r2", "conversation_id": "c2"},
        {"id": "r4", "conversation_id": "c4"},
        {"id": "r5", "conversation_id": "missing"},
        {"id": "r6"},
    ]
    apply_conversation_records(
        runs,
        {
            "c2": {"id": "c2", "title": " tagged chat ", "tags": {"archived": "true"}},
            "c4": {"id": "c4", "title": "", "tags": {}},
        },
    )

    assert runs[0]["conversation_title"] == "tagged chat"
    assert runs[0]["conversation_tags"] == {"archived": "true"}
    assert runs[1]["conversation_tags"] == {}
    assert "conversation_title" not in runs[1]
    assert "conversation_tags" not in runs[2]
    assert "conversation_tags" not in runs[3]


def test_fetch_conversation_records_batches_and_skips_a_failed_batch():
    ids = [f"c{index}" for index in range(41)]
    batch_sizes = []

    def handler(request):
        batch = request.url.params.get_list("ids")
        batch_sizes.append(len(batch))
        if len(batch) == 1:
            return httpx.Response(500, request=request)
        return httpx.Response(200, json=[{"id": item, "title": item} for item in batch], request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_conversation_records(client, "http://127.0.0.1:8010", "key", ids)

    records = asyncio.run(run())

    assert sorted(batch_sizes) == [1, 40]
    assert len(records) == 40
    assert "c40" not in records


def test_fetch_run_conversation_records_reads_the_missing_ids_from_the_ingress():
    ingress_ids = []

    def handler(request):
        ids = request.url.params.get_list("ids")
        if request.url.port == 8010:
            return httpx.Response(
                200,
                json=[{"id": "c1", "title": "run"}, None],
                request=request,
            )
        ingress_ids.extend(ids)
        return httpx.Response(200, json=[{"id": "c2", "title": "ingress"}], request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_run_conversation_records(
                client,
                "http://127.0.0.1:8010",
                "http://127.0.0.1:8000",
                "key",
                ["c1", "c2"],
            )

    records = asyncio.run(run())

    assert set(records) == {"c1", "c2"}
    assert ingress_ids == ["c2"]


def test_fetch_run_conversation_records_skips_the_second_read_on_one_backend():
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(200, json=[{"id": "c1"}], request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_run_conversation_records(
                client,
                "http://127.0.0.1:8000",
                "http://127.0.0.1:8000",
                "key",
                ["c1"],
            )

    assert set(asyncio.run(run())) == {"c1"}
    assert len(requests) == 1