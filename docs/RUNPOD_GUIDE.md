# Renting a GPU on RunPod for testing (simple steps)

Your laptop GPU (GTX 1050) is too weak for the AI avatar, so we test on a rented GPU.
You pay only while the GPU is running, about $0.35–0.90 per hour for an RTX 4090 and more for
a 5090. **Always stop it when you finish.**

## One-time setup (about 10 minutes)

1. Go to https://www.runpod.io and create an account.
2. **Billing** → add credit: $10 is enough for many test sessions.
3. **Settings → SSH Public Keys**: we'll add your key here the first time (I'll give you the
   exact command when we need it).
4. **Settings → API Keys** → *Create API Key*. Copy it and keep it private. Never paste it into
   chat, a file in the repo, or a screenshot.
5. **Set a spending limit** in Billing, so a forgotten pod can never cost much.

## Each test session

1. **Pods → Deploy**.
2. Pick the GPU I name for that test (usually **RTX 4090**, choose "Community Cloud" because it's
   cheaper).
3. Template: **RunPod PyTorch** (the newest version listed).
4. Disk: container 30 GB, volume 50 GB (unless I tell you otherwise).
5. Click **Deploy On-Demand**. Wait until the status is *Running*.
6. Click **Connect** → copy the SSH command and give it to me (or run the setup command I give
   you in the pod's web terminal).
7. When the test is done: **Stop** the pod, or **Terminate** it if we won't need its files again.

## Safety net: the pod stops itself

Every test script I give you starts with a self-destruct timer. It shuts the pod down after the
time limit (default 60 minutes), even if you forget:

```bash
# first command in every session (60 = minutes)
nohup bash -c 'sleep $((60*60)); runpodctl stop pod $RUNPOD_POD_ID' >/dev/null 2>&1 &
```

## Checklist after every session

- [ ] Pods page shows **no pod in "Running" state**
- [ ] Note the cost shown on the Billing page
