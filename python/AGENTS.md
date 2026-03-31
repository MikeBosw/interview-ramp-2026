# AGENTS.md

## Important Notes

- ALWAYS review this file before making any changes to the codebase, no matter how small the change
- The venv used by this project, if any, should be activated when making changes
- After making changes, run `git add` on the changed files, then `pre-commit run` to make sure types are checked, etc.
- Check for a PROJECT.md file adjacent to this file in case there are project-specific guidelines to follow.

## Coding Guidelines

### Python-Specific Guidelines

- We use Python for its maturity, the ecosystem, its familiarity to LLMs, its readability, and its other virtues. But we
  hate its lack of typing. We hate hate hate it. We will do everything in our power to avoid using the Any type, and to
  make the type system work for us rather than letting things crash at runtime due to an int that was actually a str. We
  will create first-class types for the data we deal with regularly. We'll do better than just a big ol' bag of dicts.
  Perhaps we'll go so far as to wrap the `int` or `str` classes because we're sick of confusing Signal ids with Telegram
  ids. If you see the opportunity to use named types to make the code safer and more self-documenting, take it. "Minor"
  typing improvements that increase the PR size by up to ~30 lines are NOT out of scope. Major typing improvements that
  increase the PR size by like, I dunno, 30%? In scope! Bless you and bless your heart.

### Design Considerations

- When first implementing something, be minimalist. "As simple as possible, but no simpler." Do not add bells or
  whistles like --dry-run or --verbose or --verify-output unless specifically asked. Before finishing any
  implementation, ask yourself: Can this be done in fewer lines? Are these abstractions earning their complexity? Are
  they at the right level and in the right place? Would a senior dev look at this and say "why didn't you just..."?
  Are there already built helper methods that you could use instead of implementing helper logic from scratch? If you
  build 1000 lines and 100 would suffice, that is a failure mode. Prefer boring, obvious solutions that do the trick.
- Treat every function signature as a design to critique, not a constraint to accept. Before writing or finalising
  any non-trivial signature, ask: can any parameter be eliminated? A parameter that shouldn't exist usually shows one
  of three symptoms:
    - **Echo parameter**: its value is always derivable from the call site (e.g. a string that always equals the
      calling method's name). Eliminate it and derive the value inside the function instead (e.g. via
      `inspect.currentframe()`).
    - **Hidden dependency**: it's optional but constructs its own fallback internally (`foo: Foo | None = None` with
      `foo = foo or Foo()` in the body). Make it required and push construction to the caller — `main()` is the right
      place for that.
    - **Parallel redundancy**: two parameters have identical signatures but differ only in which instance of a shared
      base type they operate on (e.g. `call_official: Callable[[], T]` and `call_unofficial: Callable[[], T]`).
      Collapse them into one callable parameterised on the instance (`call_with: Callable[[Base], T]`).
- Try hard to avoid state management and statefulness. If a solution exists that avoids maintaining state between
  function calls, prefer that solution. For async or parallel code, statefulness is all the more expensive.
- Strongly favor immutability by default:
    - Never re-assign a value to a variable after its initial assignment. Find other solutions.
    - Favor list comprehensions, map/reduce, and functional programming.
    - Plain dataclasses should always have frozen=True when possible, and set the equivalent Config for Pydantic models.

### Code Style

#### Errors

- Strongly prefer to throw, rather than suppress, unexpected or locally irrecoverable errors. Let it bubble up so that
  callers have the opportunity (though not the responsibility) to handle them. If the program crashes because nobody
  handled the error, that's often correct behavior. It's much easier to debug a crash due to unexpected or irrecoverable
  conditions than to debug a program pretending everything's ok when in fact things have gone awry.
- Strongly prefer throwing errors over defensively programming against unexpected values, unexpected state, or
  unexpected inputs. For example, if a parsed JSON value should be an int, or at least convertible to an int, simply
  attempt the cast. Don't check to see if it's a valid int; Python will do that for us and throw an error as
  appropriate. Likewise, if a field is always expected to be present, use `foo["bar"]` instead of `foo.get("bar")`.
- Do not wrap and rethrow errors unless there is genuinely useful, local context to be added. Instead, just throw the
  original error. If it's a specific, known error, consider making mention of it in the function's docstring.
- Avoid using None to represent an error condition. For example, bad input params, missing files, missing fields in
  files, disk OOM, and other such problems should almost always result in a thrown error, not a returned None.
  Generally, only return None when None is a valid value.
- Avoid returning True/False for success/failure. Throw errors.

#### Typing

- Don't cheat the type system. Fix the underlying problem.
- Putting quotes around the type is cheating the type system. Don't do it unless you're sure there's no alternative.
- You have permission to go to great lengths to avoid using the Any type. Evaluate the structure of JSON blobs and type
  them using things like TypedDict.
- Whenever possible, use native types (list, dict) for type hints rather than importing from typing (List, Dict, etc).
- Never do lazy imports as a "fix" for circular imports. Fix them for real. Propose a refactor that makes sense without
  creating a ton of spaghetti code. It's hard, but circular imports are completely unacceptable and we use types for a
  reason.
- Do not include imports inside of a function body.
- Do not make complex things happen at import time. Delay until time of usage.
- Never do lazy imports if real types will work.
- If there is absolutely no way to make the type system work as desired, ALWAYS justify the hacky workaround with a
  comment explaining why it was necessary.
- When passing callables to APIs that erase argument types (e.g. executor.map, executor.submit, Thread(target=...)),
  create named zero-argument functions that capture the args in their body. This ensures the type checker validates the
  arguments at the call site inside the function, rather than losing them to Callable[..., T]. For APIs that preserve
  argument types (e.g. sorted(key=...), map()), inline lambdas or references are fine.

#### Everything Else

- Highly readable lines of code must not be commented. Comments should exist only to explain opaque logic, missing
  business/operational context, the justification for hacky workarounds, quirks in third-party behavior that have to be
  accommodated, or code with middling to low readability. If there's a `json.loads` call, for example, you should
  absolutely not comment that you're loading a JSON string. Likewise, there should absolutely not be a comment
  `# send result to user` for an invocation of `send_to_user`.
- On the flip side, ALL pyright / mypy / tsc type errors must be commented. You should either explain why the type
  system is sufficiently deficient to justify an override, or else fix the type error.
- Log messages should be no more than one sentence long. The first letter should not be capitalized unless it's an
  acronym or proper noun. The message should not end with a period.
- Avoid empty lines of whitespace unless required by PEP / Ruff / Black style requirements.
- Pass the pre-commit hooks or modify and re-commit
- Minimize the scope of try blocks. Prefer to do only the exception-prone logic within the try block, rather than all
  the before and after logic.
- Minimize both indentation and loss of context by putting abortive conditions into standalone if-then-return statements
  at the top of the function or loop. For example, this is an anti-pattern:
  ```python
    if conversations:
      print("\nFirst conversation details:")
      # ... a whole bunch of logic
    else:
      print("No conversations found")
  ```
  Instead, do this:
  ```python
    if not conversations:
      print("No conversations found")
      return  # or break, or continue
    print("\nFirst conversation details:")
    # ... a whole bunch of logic
  ```
- The first line of a Git commit message should begin with a lowercase letter (unless it's an acronym or proper noun),
  be concise, and end without a period
- Function docs use the present tense in the indicative voice and skip the subject, e.g., "Creates a new user."
- Class docs typically begin with "A" or "The" followed by a noun phrase describing what the entity is, followed by (if
  more than just a simple data object) what the entity does.
- Minimize the scope of variables to the smallest possible context. If something is only used within the else clause of
  an if-then-else, for example, declare it in the else clause, not before the if.
- Favor named enums over raw ints or strings
- Extrapolate equivalent guidance, based on the above, to other languages
- Whenever it's possible to use f-strings for log messages, never use the %d %s %f stuff.

## PRs

- When implementing PRs, do so on a feature/ branch and submit the PR on that branch
- When reacting to PR feedback, post an inline reply to every comment. The reply should appear not at the PR level but
  at the level of the comment thread.
- When posting replies to comments, preface replies with `[agent-name]`; for example, if you're Claude Code, preface the
  replies each with `[claude]`.
- When something is valid but out of scope, add a `TODO(agent-name): what to do or consider doing, and why` comment.
- Incidental refactoring is preferable if it's not large-scale. An example would be moving a private constant previously
  used by a single file into a shared place if it becomes useful to multiple files.
- Before implementing a PR, consider which parts of the logic would likely exist elsewhere, then look for those bits,
  then make sure to reuse them rather than re-implementing the same logic. If you find a chunk of logic that would be
  useful in multiple places, consider whether it makes sense to refactor it, or just accept a smidge of duplication.
- Do not use existing code as an excuse to perpetuate bad patterns. If a bug has been verified in the new code, fix it,
  even if the same bug exists in old code. Then add a TODO(agent-name) anywhere that the bug appears in the old code but
  that you deem out of scope for current PR.

## Attribution

https://raw.githubusercontent.com/MikeBosw/agents-guidance/refs/heads/main/AGENTS.md
