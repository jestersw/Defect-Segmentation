# ResNet18 Dice experiment at 384

Configuration: `configs/cpu_dice_384.yaml`. Run: `cpu_dice_384`.

This jointly changes input resolution (256 to 384), decoder channels
([128,64,32,16,8] to [256,128,64,32,16]), and BCE to pure SMP binary Dice
loss. It cannot isolate the contribution of each change. Training starts
from ImageNet encoder weights, not the previous checkpoint. Validation,
splits, seed, batch size and optimizer remain the same. Select by validation
Dice and inspect small recall and good-image false positives as well.

The previous run remains in `results/runs/cpu_light_bce_256` and its weights
in `checkpoints/cpu_light_bce_256`. Best validation Dice: 0.70852454, epoch 58.

Training stops at 60 epochs, patience 10, or the time budget. The 17,100-second
budget is checked between epochs using the slowest observed epoch plus 25%
reserve. This targets less than five hours but is not a hard wall-clock limit:
unexpected machine slowdown can exceed the estimate. Best and last completed
epoch checkpoints are preserved when stopping for time.

Background launch (PowerShell):

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File scripts/start_training.ps1 -Config configs/cpu_dice_384.yaml -RunName cpu_dice_384
```

Monitor in Git Bash:

```bash
tail -f .tools/training/cpu_dice_384/stdout.log
```

Errors: `.tools/training/cpu_dice_384/stderr.log`.
Progress and final status: `results/runs/cpu_dice_384/summary.json`.
Best weights: `checkpoints/cpu_dice_384/best.pth`.
