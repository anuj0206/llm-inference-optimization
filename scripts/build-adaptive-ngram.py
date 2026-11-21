"""Build the alternate host DLL using the pinned CUDA build's unchanged objects."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

PIN = "8bf88ba9368e70fe3ba1b0001b34e70ffc713716"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError(f"Unexpected stock build or source layout around {before!r}")
    return text.replace(before, after)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    project = Path(__file__).resolve().parent.parent
    root = Path(os.environ["LLM_OPT_ROOT"])
    source = root / "llama.cpp"
    stock = root / "build/llama-cuda"
    target = root / "build/llama-adaptive-ngram"
    binary = target / "bin"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if revision != PIN or dirty:
        raise RuntimeError("The clean pinned llama.cpp source is required.")

    database = json.loads(subprocess.check_output([
        "ninja", "-t", "compdb", "-x", "CXX_COMPILER__llama-common_unscanned_Release",
        "CXX_SHARED_LIBRARY_LINKER__llama-common_Release",
    ], cwd=stock, text=True))
    compiler = next(entry for entry in database if entry["output"].endswith("/ngram-map.cpp.obj"))
    linker = next(entry for entry in database if entry["output"] == "bin/llama-common.dll")
    binary.mkdir(parents=True, exist_ok=True)
    manifest = target / "build.json"
    manifest.unlink(missing_ok=True)
    policy = project / "benchmarks/ngram/consensus-backoff.inc"
    ranked = policy.with_name("ranked-draft.inc")
    shutil.copy2(policy, target / policy.name)
    shutil.copy2(ranked, target / ranked.name)
    source_text = (source / "common/ngram-map.cpp").read_text(encoding="utf-8")
    source_text = replace_once(source_text, '#include "ngram-map.h"',
                               '#include "ngram-map.h"\n#include "consensus-backoff.inc"\n#include "ranked-draft.inc"')
    source_text = replace_once(source_text, "    // Simple implementation of self-speculative decoding",
                               """    static const ngram_experiment experiment;
    if (experiment.probe_length) {
        return llama_tokens(experiment.probe_length, sampled);
    }
    if (experiment.ranked) {
        return ngram_ranked_draft(experiment, tokens, sampled);
    }

    // Simple implementation of self-speculative decoding""")
    for condition in (
        "cur_len <= static_cast<size_t>(n_draft_min + n_draft_max + 1)",
        "match_pos == 0",
        "copy_max < n_draft_min",
    ):
        before = f"if ({condition}) {{\n        return draft_tokens;"
        after = f"if ({condition}) {{\n        return ngram_consensus_backoff(config, tokens, sampled);"
        source_text = replace_once(source_text, before, after)
    candidate_source = target / "ngram-map.cpp"
    candidate_source.write_text(source_text, encoding="utf-8")

    stock_object = str(Path(compiler["output"]))
    candidate_object = target / "ngram-map.cpp.obj"
    command, separator, _ = compiler["command"].rpartition(" -c ")
    if not separator:
        raise RuntimeError("Compiler command is missing the source argument.")
    command = replace_once(command, f"/Fo{stock_object}", f'/Fo"{candidate_object}"')
    command = replace_once(command, "/Fdcommon\\CMakeFiles\\llama-common.dir\\", f'/Fd"{target / "ngram-compile.pdb"}"')
    command = command.replace(" /showIncludes ", " ")
    subprocess.run(f'{command} -c "{candidate_source}"', cwd=stock, check=True)

    match = re.search(r' -- "([^"]+\\link\.exe)" (.+) && cd \."', linker["command"])
    if not match:
        raise RuntimeError("Could not locate the linker invocation in the stock build.")
    link_exe, link_args = match.groups()
    for original, replacement in (
        (stock_object, str(candidate_object)),
        ("bin\\llama-common.dll", str(binary / "llama-common.dll")),
        ("common\\llama-common.lib", str(target / "llama-common.lib")),
        ("bin\\llama-common.pdb", str(target / "llama-common.pdb")),
    ):
        link_args = replace_once(link_args, original, f'"{replacement}"')
    response = target / "link.rsp"
    response.write_text(link_args + "\n", encoding="utf-16")
    subprocess.run([link_exe, f"@{response}"], cwd=stock, check=True)

    stock_bin = stock / "bin"
    for dll in stock_bin.glob("*.dll"):
        if dll.name != "llama-common.dll":
            shutil.copy2(dll, binary / dll.name)
    for name in ("llama-cli.exe", "llama-server.exe"):
        shutil.copy2(stock_bin / name, binary / name)
    manifest.write_text(json.dumps({
        "source_commit": revision,
        "policy_sha256": sha256(policy),
        "ranked_policy_sha256": sha256(ranked),
        "stock_common_sha256": sha256(stock_bin / "llama-common.dll"),
        "adaptive_common_sha256": sha256(binary / "llama-common.dll"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Adaptive n-gram binaries ready at {binary}")


if __name__ == "__main__":
    main()
