import asyncio
import json
from typing import cast

from httpx import AsyncClient, HTTPStatusError

# everything is of the form GET domain /id (fail if not)
URL = "https://bags-volvo-edition-usc.trycloudflare.com"


EXAMPLE = """
{
    "next_steps": [
        "TVAUJPABEY",
        "FTYYIJNUCT"
    ]
}
"""


CONGRATS = "Congrats"

MAX_RETRIES = 10


class Crawly:
    def __init__(self) -> None:
        self.client = AsyncClient()

    async def fetch_next_steps(self, step_id: str) -> list[str] | str:
        assert "/" not in step_id
        backoff = 0.1
        for i in range(MAX_RETRIES):
            await asyncio.sleep(backoff)
            res = await self.client.get(f"{URL}/{step_id}")
            if res.status_code != 200:
                print(f"failing step {step_id}")
            if res.status_code not in [200, 503]:
                print(f"non-503 error for step {step_id}: {res.status_code}")
            if res.status_code == 200:
                break
        res.raise_for_status()
        data = res.read()
        text = data.decode("UTF-8")
        if "CONGRATS" in text:
            return CONGRATS
        blob = json.loads(text)
        next_steps = cast(list[str], blob["next_steps"])
        return next_steps

    async def crawl(self) -> str:
        queue = [""]
        visited_step_ids: set[str] = set()
        failed_steps: set[str] = set()
        while queue:
            all_new_steps: set[str] = set()
            for step_id in queue:
                if step_id in visited_step_ids:
                    # breaks if steps change under us
                    continue
                visited_step_ids.add(step_id)
                try:
                    new_steps: list[str] | str = await self.fetch_next_steps(step_id)
                except HTTPStatusError:
                    print(f"failed to take step {step_id}")
                    failed_steps.add(step_id)
                    continue
                if not isinstance(new_steps, list):
                    return step_id  # <-- break first then return maybe?
                all_new_steps.update([s for s in new_steps if s not in visited_step_ids])
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
