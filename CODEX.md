
Write an Architecture.md for this repo.  

See this link for how to create an Architecture.md file that describe system architecture (https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html).  

I'm approaching this repo as a developer who wants to understand its internals:
- how data flows from user facing APIs through its various components
- the path from user facing APIs through the various abstraction layers to actual host / device cuda kernel calls
- how multi-device synchronization is orchestrated

The idea is to provide "literate code": a guided line-by-line walkthrough of the call stack / data flow through inline code snippets and mappings to corresponding code sections (source file + line number spans).  Think explanatory prose + code + clickable links + visuals.

Spare no details -- I want to understand the entire call path from **both** the C++ AND Python APIs.

As a supplement to the ARCHITECTURE.md, provide annotated walkthroughs of the tests (both intra and internode):
- start from the user facing API
- fully unfurl the entire call path for these APIs down to the lowest level: inline code snippets + code links for each function / method along each call path.
- trace from python -> bindings -> wrappers -> C++ -> cuda driver API / kernel calls

For each trace, provide annotated inline code snippets along with source code and line number links to relevant sections to help build a visual map of the entire call stack.  Ensure that these links are markdown style links and clickable from with VSCode, e.g.,
[barrier.cuh:196](src/include/non_abi/device/coll/barrier.cuh#L196)

Liberal use of markdown visuals (tables, diagrams, etc.) is encouraged to help illustrate.

Assume minimal background with NVLink, Infiniband, RDMA, and other CUDA networking technologies: make sure to include background primers alongside the code walkthroughs to help illuminate not just the how but the why of what the code is doing.
 
Do all your work in a folder "codex" at the repo root.
