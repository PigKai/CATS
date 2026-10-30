set -euo pipefail
exec bash "$(dirname "$0")/train_llava_endoSpec.bash" "$@"
