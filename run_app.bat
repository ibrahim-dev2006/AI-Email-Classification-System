@echo off
REM Double-click this file to start Winnow.
cd /d "%~dp0"

echo ============================================================
echo   Winnow - AI Email Classification System
echo ============================================================
echo.
echo   Opening http://localhost:8501 in your browser.
echo   Sign in with:  admin  /  ChangeMe!123
echo.
echo   Keep this window open while you use the app.
echo   Press Ctrl+C here to stop the server.
echo.
echo ============================================================
echo.

python -m streamlit run app.py --server.port 8501

if errorlevel 1 (
    echo.
    echo The server stopped with an error. Read the message above.
    echo If it says a module is missing, run:  pip install -r requirements.txt
    echo.
    pause
)
