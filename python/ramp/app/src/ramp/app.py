import asyncio
import json
from typing import cast

from httpx import AsyncClient


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


class Crawly:
    def __init__(self) -> None:
        self.client = AsyncClient()

    async def fetch_next_steps(self, step_id: str) -> list[str] | str:
        assert "/" not in step_id
        if "CONGRATS" in step_id:
            return CONGRATS
        res = await self.client.get(f"{URL}/{step_id}")
        data = res.read()
        text = data.decode("UTF-8")
        blob = json.loads(text)
        next_steps = cast(list[str], blob["next_steps"])
        return next_steps

    async def crawl(self) -> str:
        queue = [""]
        visited_step_ids: set[str] = set()
        while queue:
            all_new_steps: set[str] = set()
            for step_id in queue:
                if step_id in visited_step_ids:
                    # breaks if steps change under us
                    continue
                new_steps: list[str] | str = await self.fetch_next_steps(step_id)
                visited_step_ids.add(step_id)
                if not isinstance(new_steps, list):
                    return step_id  # <-- break first then return maybe?
                all_new_steps.update([s for s in new_steps if s not in visited_step_ids])
            queue = [s for s in all_new_steps]
        raise RuntimeError("maze has no exit")


async def main() -> None:
    crawly = Crawly()
    winning_step_id = await crawly.crawl()
    print(winning_step_id)


if __name__ == "__main__":
    asyncio.run(main())
