import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from httpx import AsyncClient, HTTPStatusError, Response

# everything is of the form GET domain /id (fail if not)
URL = "https://bags-volvo-edition-usc.trycloudflare.com/b-side"

EXAMPLE = """
{
    "next_steps": [
        "TVAUJPABEY",
        "FTYYIJNUCT"
    ]
}
"""

CONGRATS = "Congrats"

MAX_ATTEMPTS = 10


class KeyNeededException(Exception):
    def __init__(self, key: str):
        self.key = key


@dataclass
class NextSteps:
    step_ids: list[str]
    findings: dict[str, str]


class Crawly:
    def __init__(self) -> None:
        self.client = AsyncClient()

    async def fetch_next_steps(self, step_id: str, keys: dict[str, str]) -> NextSteps | str:
        assert "/" not in step_id
        backoff = 0.1
        key_required: str | None = None
        for i in range(MAX_ATTEMPTS):
            await asyncio.sleep(backoff)
            headers = {key_required: keys[key_required]} if key_required and key_required in keys else None
            res: Response = await self.client.get(f"{URL}/{step_id}", headers=headers)
            if res.status_code in [404, 200]:
                break
            if res.status_code == 401:
                text = res.read().decode("UTF-8")
                key_name = json.loads(text)["header"]
                if key_name in keys:
                    key_required = key_name
                    continue
                else:
                    raise KeyNeededException(key_name)
            if res.status_code != 200:
                print(f"failing step {step_id}")
            if res.status_code not in [503]:
                print(f"non-503 error for step {step_id}: {res.status_code}")
        res.raise_for_status()
        data = res.read()
        text = data.decode("UTF-8")
        if "CONGRATS" in text:
            return CONGRATS
        blob = json.loads(text)
        next_step_ids = cast(list[str], blob["next_steps"])
        findings = {}
        if "key_name" in blob:
            findings[blob["key_name"]] = blob["key_value"]
            print(f"found key: {findings}")
        return NextSteps(next_step_ids, findings)

    async def crawl(self) -> str:
        queue = [""]
        visited_step_ids: set[str] = set()
        failed_steps: set[str] = set()
        keys_needed: dict[str, list[str]] = defaultdict(lambda: [])
        keys_found: dict[str, str] = {}
        while queue:
            all_new_steps: set[str] = set()
            for step_id in queue:
                if step_id in visited_step_ids:
                    # breaks if steps change under us
                    continue
                visited_step_ids.add(step_id)
                try:
                    new_steps: NextSteps | str = await self.fetch_next_steps(step_id, keys_found)
                except HTTPStatusError:
                    print(f"failed to take step {step_id}")
                    failed_steps.add(step_id)
                    continue
                except KeyNeededException as e:
                    print(f"found locked door at step {step_id}, need key {e.key}")
                    keys_needed[e.key].append(step_id)
                    continue
                if not isinstance(new_steps, NextSteps):
                    return step_id  # <-- break first then return maybe?
                for key in new_steps.findings:
                    for step in keys_needed[key]:
                        visited_step_ids.remove(step)
                        queue.append(step)
                keys_found.update(new_steps.findings)
                all_new_steps.update([s for s in new_steps.step_ids if s not in visited_step_ids])
            queue = [s for s in all_new_steps]
        raise RuntimeError(
            f"maze has no exit: could be one of the {len(failed_steps)}/{len(visited_step_ids)} failed steps?"
        )


async def main() -> None:
    crawly = Crawly()
    winning_step_id = await crawly.crawl()
    print(winning_step_id)


if __name__ == "__main__":
    asyncio.run(main())
