```bash
PS C:\Users\konos\DocLayout-YOLO> .\venv\Scripts\Activate
(venv) PS C:\Users\konos\DocLayout-YOLO>
(venv) PS C:\Users\konos\DocLayout-YOLO> python demo_keep.py --model weights/yolov10-doclayout.pt --image-path assets/example/* --imgsz 1024 --conf 0.35 --keep-class abandon --save-json
```

preview

```pwsh
Get-ChildItem -Path . -File | Where-Object { $_.Name -notlike 'crop*' }
```

**Delete files that do not start with crop**

```pwsh
Get-ChildItem -Path . -File | Where-Object { $_.Name -notlike 'crop*' } | Remove-Item -Force
```

(venv) PS C:\Users\konos\DocLayout-YOLO> python rotate_odd_even.py --dir assets\example --inplace
