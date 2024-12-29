echo "****************************************************"
echo "****                                            ****"
echo "****    ---- Starting the Kaggle Team ----      ****"
echo "****                                            ****"
echo "****************************************************"
echo "****                                            ****"
echo "****    Welcome to the Grand Kaggle Setup!      ****"
echo "****    Prepare for an Exciting Journey!        ****"
echo "****                                            ****"
echo "****************************************************"

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'micromamba shell init' !!
export MAMBA_EXE='/root/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/root/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<

apt-get install vim -y
apt-get install htop -y
apt-get install build-essential -y
pip install uv
uv pip install \
    --no-binary lightgbm \
    --config-settings=cmake.define.USE_CUDA=ON \
    lightgbm
uv pip install protobuf
uv pip install grpcio

echo "****************************************************"
echo "****                                            ****"
echo "****    ---- Please Press Enter 4 Times ----    ****"
echo "****                                            ****"
echo "****************************************************"

"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source /root/.bashrc

micromamba create --name kaggle python=3.11 -c conda-forge -y
micromamba activate kaggle 
micromamba config append channels conda-forge
micromamba config append channels torch 
# linux-cuda 12.4
micromamba install --yes -c conda-forge boost

micromamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -y -c pytorch -c nvidia 
micromamba install polars numpy xgboost lightgbm dill matplotlib optuna kaggle pandas pyarrow fastparquet catboost -y -c conda-forge 
micromamba install conda-forge::grpc-cpp
kaggle competitions download -c jane-street-real-time-market-data-forecasting
unzip jane-street-real-time-market-data-forecasting.zip 'kaggle_evaluation/*'
mkdir -p 
unzip jane-street-real-time-market-data-forecasting.zip -d /kaggle/input/jane-street-realtime-marketdata-forecasting