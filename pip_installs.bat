@echo off
REM Install the required packages in the current Python environment
pip install keras==2.12.0 urllib3==1.26.14 yfinance==0.2.50 requests==2.31.0 scikit-learn==1.5.2 ta==0.11.0 alpaca-trade-api==3.2.0 textblob==0.18.0 pycoingecko==3.2.0 python-binance==1.0.22 python-decouple==3.5.0 pyportfolioopt==1.5.5 pandas==1.5.3 numpy==1.23.5 scipy==1.10.0 tensorflow==2.12.0 matplotlib==3.7.3 requests

REM Indicate completion
echo Packages installed successfully.
pause