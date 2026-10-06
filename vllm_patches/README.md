# vLLM patches

This directory contains the entropy-aware decoding patch used by `generate_rollouts.sh`.

The patch is generated against vLLM `v0.15.1` (`1892993bc`). At job startup, the script applies `v0.15.1-entropy-aware-decoding.patch` to the installed vLLM package. By default that target is:

```bash
/usr/local/lib/python3.12/dist-packages/vllm
```

Set `VLLM_SITE_ROOT=/path/to/site-packages/vllm` if the container installs vLLM somewhere else.
