GPU=$1

# Define corresponding output names and corruption values in order.
corruptions=(
    ""              # clean
    "NoImage"       # noimage
    "BitError"      # biterror
    "CameraCrash"   # cameracrash
    "Fog"           # fog
    "H256ABRCompression"  # h256
    "LowLight"      # lowlight
    "Rain"          # rain
    "Snow"          # snow
    "Brightness"    # bright
    "ColorQuant"    # colorquant
    "FrameLost"     # framelost
    "LensObstacleCorruption"  # lens
    "MotionBlur"    # motion
    "Saturate"      # saturate
    "ZoomBlur"      # zoom
    "WaterSplashCorruption"   # water
)

outputs=(
    "clean"
    "noimage"
    "biterror"
    "cameracrash"
    "fog"
    "h256"
    "lowlight"
    "rain"
    "snow"
    "bright"
    "colorquant"
    "framelost"
    "lens"
    "motion"
    "saturate"
    "zoom"
    "water"
)
MODEL_THINKING="/scratch/gpfs/ZHUANGL/el5267/cache/huggingface/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/41ea130ce6eaaf7829c72dfc0e4597d49741ed18"

# Loop over the arrays and run the commands.
for i in "${!outputs[@]}"; do
    OUTPUT_FILE="res/qwen3-vl-8b-thinking/${outputs[i]}.json"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping ${outputs[i]} — $OUTPUT_FILE already exists"
        continue
    fi
    python inference/qwen3.py \
        --model "$MODEL_THINKING" \
        --data data/drivebench-test-final.json \
        --output "res/qwen3-vl-8b-thinking/${outputs[i]}" \
        --system_prompt prompt.txt \
        --num_processes "${GPU}" \
        --max_model_len 32768 \
        --max_tokens 4096 \
        --thinking \
        --corruption "${corruptions[i]}"
done
