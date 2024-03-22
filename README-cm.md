# MLCommons CM automation for ABTF

Install [MLCommons CM automation meta-framework](https://access.cknowledge.org/playground/?action=install).

Pull [main automation recipes](https://access.cknowledge.org/playground/?action=scripts) from MLCommons:

```bash
cm pull repo mlcommons@ck --checkout=dev
```

Pull repository with CM scripts for MLCommons-ABTF benchmark:

```bash
cm pull repo mlcommons@abtf-ssd-pytorch --checkout=cognata-cm
```

Clean CM cache if you want to start from scratch

```bash
cm rm cache -f
```

