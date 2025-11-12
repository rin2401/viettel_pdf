curl -LsSf https://astral.sh/uv/install.sh | sh

pwd=$(pwd)

echo "Extract"
cd $pwd/extract && uv sync

echo "QA"
cd $pwd/qa && uv sync
