import asyncio

from httpx import AsyncClient


async def main() -> None:
    res = await AsyncClient().get("https://www.xkcd.com/")
    print(res.text)


if __name__ == "__main__":
    asyncio.run(main())
