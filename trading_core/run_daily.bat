@echo off
REM ============================================================
REM  trading_core - 日次ペーパートレード実行 (ダブルクリック / スタートアップ両対応)
REM  この .bat 自身の場所へ移動してから uv を実行するので、
REM  どこから呼ばれても正しく動く (%~dp0 = このファイルのフォルダ)。
REM ============================================================
cd /d "%~dp0"
echo === trading_core daily run  %date% %time% ===

where uv >nul 2>&1
if %errorlevel%==0 (
    uv run python -m execution.live_runner run --mode paper
) else if exist "%USERPROFILE%\.local\bin\uv.exe" (
    "%USERPROFILE%\.local\bin\uv.exe" run python -m execution.live_runner run --mode paper
) else (
    echo *** uv が見つかりません。uv のインストールを確認してください ***
    pause
    exit /b 1
)

if %errorlevel% neq 0 (
    echo.
    echo *** 実行に失敗しました。上の表示を確認してください ***
    pause
)
