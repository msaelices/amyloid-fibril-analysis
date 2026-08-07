# data/

This folder is **gitignored** — never commit patient micrographs or annotations.

Suggested layout:

```
data/
  raw/          # original .mrc micrographs, e.g. P1/img001.mrc
  annotations/  # manual traces: ImageJ RoiSet.zip / .roi, or CSV (filament_id,x,y)
  processed/    # derived masks, probability maps, cached traces
```

Organize by patient (`P1/`, `P2/`, `P3/`) so patient ids are unambiguous.
