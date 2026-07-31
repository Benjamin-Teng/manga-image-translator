# 原生 venv (uv) 建置與運作紀錄

> 日期：2026-06-26
> 目的：把漫畫翻譯從 Docker/WSL2 路徑改成「原生 Windows venv 直接吃 GPU」，
> 移除 WSL2 / Docker / GPU-PV 這一層（見 `crash-report.md`：外接 4K@240 + 翻譯時
> 的顯示路徑死結／除零當機；原生路徑是該問題的對照實驗 + 緩解方案）。
> 結論：**可行，且與 Docker 行為一致**，但有一個 Windows 專屬地雷（cp950，見下）。

---

## 0. 環境前提（本機實測值）

| 項目 | 值 / 說明 |
|---|---|
| OS | Windows 11 (26200) |
| GPU | RTX 5090 Laptop GPU（Blackwell, **sm_120**）|
| 驅動 | 610.47（注意：此版對外接螢幕更不穩，見 crash-report.md §11.6）|
| 系統 Python | 3.13.7（**太新**，專案要求 `>=3.10,<3.12`，不能直接用）|
| C++ 編譯器 | Visual Studio Community 2022（`pydensecrf` 要從原始碼編，需要它）|

---

## 1. 安裝 uv

```powershell
irm https://astral.sh/uv/install.ps1 | iex
# 裝到 C:\Users\<you>\.local\bin\uv.exe（本機實測 uv 0.11.24）
```

## 2. 建 venv（Python 3.11）

系統 3.13 太新，讓 uv 自己抓 3.11：

```powershell
uv python install 3.11
uv venv --python 3.11 D:\Manga-Translator\.venv      # -> Python 3.11.15
```

## 3. 裝 GPU 版 torch（cu128 / Blackwell）

對齊 `Dockerfile.cu128`：官方映像內建的 torch 只到 sm_90，認不得 sm_120，必須換 cu128。

```powershell
$py = "D:\Manga-Translator\.venv\Scripts\python.exe"
uv pip install --python $py torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 `
    --index-url https://download.pytorch.org/whl/cu128
```

驗證（要看到 `sm_120` + 認得 5090）：

```powershell
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# torch 2.8.0+cu128 True NVIDIA GeForce RTX 5090 Laptop GPU (12, 0)
```

## 4. 裝其餘依賴（torch 釘死成 cu128，避免被覆蓋）

`requirements.txt` 裡 `torch` 是裸版本，直接 `-r` 安裝會被 resolver 換成 PyPI 的 CPU 版，
所以要把 cu128 版本一起釘進去；同時補上 rust wheel 的 extra index：

```powershell
uv pip install --python $py --index-strategy unsafe-best-match `
    --extra-index-url https://download.pytorch.org/whl/cu128 `
    --extra-index-url https://frederik-uni.github.io/manga-image-translator-rust/python/wheels/simple/ `
    -r requirements.txt `
    torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128
```

實測三個「容易卡」的依賴都過了：
- `pydensecrf==1.0`：用 VS2022 從 GitHub zip **編譯成功**
- `rusty-manga-image-translator==0.12.1`：有 Windows wheel
- `bitsandbytes==0.49.2`：有 Windows wheel
- `numpy` 被正確壓回 `1.26.4`（torch 先裝的 2.4.4 會被降級，正常）

## 5. 模型

第一次跑會自動下載到 `models\`（detector 約 294MB + ocr + lama_large 等）。
不必先跑 `docker_prepare.py`（且它的 `--models` 過濾因 Enum 字串問題會比對不到，
不帶參數則會下載全部，太多；直接讓翻譯時按需下載即可）。

---

## 6. 碰到的問題（重要）：Windows cp950 編碼

**症狀**：第一次原生跑，翻譯**全部失敗**，輸出圖片沒翻到：
```
ERROR: 'cp950' codec can't encode character '込' ... illegal multibyte sequence
WARNING: ... Translation identical to original   (被當成沒翻而濾掉)
```

**原因**：Docker 跑 Linux，預設編碼 UTF-8；**原生 Windows 預設是 cp950（繁中 ANSI 碼頁）**，
送 OpenAI 請求遇到日文字（如「込」）就編不出來 → 每段重試失敗 → 靜默產出未翻譯結果。

**解法**：強制 Python UTF-8 模式
```powershell
$env:PYTHONUTF8 = "1"
```
加上後重跑，Stage 1/2 全成功，輸出正確繁體中文，與 Docker 一致。
→ **這是原生路徑的硬性條件**，已寫進 `translate.local.venv.bat`。

> 另有一個無害警告（Docker 也會有，非原生造成）：
> `Failed to generate bboxes_fixed.png: cannot import name 'visualize_textblocks'`
> 只是 debug 視覺化圖產不出來，不影響翻譯結果。

---

## 7. 跑法

手動：
```powershell
$env:PYTHONUTF8 = "1"
D:\Manga-Translator\.venv\Scripts\python.exe -m manga_translator local `
    -i input -o "result\xxx" --config-file golden_config.local.json --use-gpu
```

一鍵（推薦）：**`translate.local.venv.bat`**（行為等同 `translate.local.bat`，但走原生 venv）
- 無參數：`input\` → `result\result-YYMMDD\`
- 單一或多個資料夾：`translate.local.venv.bat "D:\...\Manga"` → `result\<grandparent>\<name>-translated\`
- 內含：自動 `PYTHONUTF8=1`、單一執行鎖、per-manga `_glossary.txt` 處理（排除翻譯 + 原地更新）、可續傳（跳過已翻）
- 注意：`*.local.*` 已被 `.gitignore`，此檔不會被推送（與 `translate.local.bat` 同慣例）

---

## 8. 驗證結果（2026-06-26 實測，皆 exit 0）

| 測試 | 結果 |
|---|---|
| torch 認 GPU | ✅ 2.8.0+cu128 / sm_120 / RTX 5090 |
| 無參數模式（`input\`）| ✅ 11.1s，正確繁中，存到 `result\result-260626\` |
| 資料夾模式（單路徑）| ✅ 輸出到 `result\<grand>\<name>-translated\`（grandparent 路徑正確）|
| `_glossary.txt` 排除 | ✅ 輸出夾只有圖片、沒有被翻譯的 glossary |
| glossary 原地更新 | ✅ 來源 `_glossary.txt` 被讀取且 `+2` 新專名寫回 |
| 暫存清理 | ✅ `%TEMP%\mit_venv_in_*` 跑完清乾淨 |
| 暖跑速度 | 約 11–13 秒/張（首跑含模型下載 ~2.5 分）|

→ **原生 venv 路徑與 Docker 行為一致，且完全不經 WSL2 / Docker / GPU-PV。**
可拿它做 crash-report.md 的對照實驗：原生 + 外接 4K@240 跑長翻譯，看是否還當。
