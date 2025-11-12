curl -LsSf https://astral.sh/uv/install.sh | sh

pwd=$(pwd)

cd $pwd/extract & uv sync
cd $pwd/qa & uv sync
