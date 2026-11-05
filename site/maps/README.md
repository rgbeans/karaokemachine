# Map packages

Put exported KaraokeMachine `.json` map packages in this folder.

Then run:

```bash
npm run maps:index
```

This generates `index.json` for the map browser. Map packages must not contain audio.
A chart with no audio source sets `"noAudio": true` and leaves `sourceUrl` empty; it is
listed with a **NO AUDIO** label. See the repository's `CONTRIBUTING.md` for the full
checklist.
