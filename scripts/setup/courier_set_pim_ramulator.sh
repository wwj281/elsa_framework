## initialize ramulator2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PATCH_SRC="${PROJECT_ROOT}/ramulator_patches/courier_pim_ramulator_src"
cd "${PROJECT_ROOT}" || exit 1

cd ramulator2
git reset --hard b7c70275f04126c647edb989270cc429776955d1
cd ..

cd "${PATCH_SRC}/patches/"
unix2dos ./*
cd "${PROJECT_ROOT}" || exit 1

# copy files
cp "${PATCH_SRC}/DDR4_PIM.cpp" ramulator2/src/dram/impl/
cp "${PATCH_SRC}/LPDDR5_PIM.cpp" ramulator2/src/dram/impl/
cp "${PATCH_SRC}/ddr_pim_controller.cpp" ramulator2/src/dram_controller/impl/
cp "${PATCH_SRC}/lpddr_pim_controller.cpp" ramulator2/src/dram_controller/impl/
cp "${PATCH_SRC}/pim_scheduler.cpp" ramulator2/src/dram_controller/impl/scheduler/
cp "${PATCH_SRC}/pim_loadstore_trace.cpp" ramulator2/src/frontend/impl/memory_trace/
cp "${PATCH_SRC}/PIM_DRAM_system.cpp" ramulator2/src/memory_system/impl/
cp -r "${PATCH_SRC}/trace_gen" ramulator2/
cp -r "${PATCH_SRC}/patches" ramulator2/

# Apply patches
cd ramulator2;

for f in ./patches/*.patch
do
    patch --binary --ignore-whitespace -p1 < $f
done
