# Google Earth Engine Authentication Guide

Since you are running the application locally, you need to authenticate with Google Earth Engine (GEE) to allow the Python script to fetch satellite data.

## Prerequisites
1.  **Google Cloud Project**: You need a Google Cloud Project with the **Earth Engine API** enabled.
2.  **GEE Account**: Your Google account must be signed up for Earth Engine at [earthengine.google.com/signup](https://earthengine.google.com/signup/).

## Authentication Steps

### 1. Run the Authentication Command
Open your terminal (in the project directory) and run:

```bash
.\.venv\Scripts\earthengine.exe authenticate
```
*Note: If that doesn't work, try `.\.venv\Scripts\python.exe -m ee authenticate`*

### 2. Browser Login
1.  The command will open a URL in your default web browser (or print a URL to copy-paste).
2.  **Log in** with the Google Account associated with Earth Engine.
3.  **Allow** the Earth Engine Notebook Client to access your data.
4.  **Copy** the authorization code provided.

### 3. Complete Setup
1.  Paste the authorization code back into the terminal and press **Enter**.
2.  The tool will save your credentials to a local file (usually `%USERPROFILE%\.config\earthengine\credentials`).

## Verification
To verify it works, run the python shell:

```powershell
.\.venv\Scripts\python.exe
```

Then run:
```python
import ee
ee.Initialize()
print(ee.Image("NASA/NASADEM_HGT/001").get("system:id").getInfo())
```
If it prints `NASA/NASADEM_HGT/001`, you are successfully authenticated!
