## initialize ramulator2 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PATCH_SRC="${PROJECT_ROOT}/ramulator_patches/pim_ramulator_src"
cd "${PROJECT_ROOT}" || exit 1

cd ramulator2
git reset --hard b7c70275f04126c647edb989270cc429776955d1
cd ..

# copy files
cp "${PATCH_SRC}/attacc_bank.yaml" ramulator2/
cp "${PATCH_SRC}/attacc_bg.yaml" ramulator2/
cp "${PATCH_SRC}/attacc_buffer.yaml" ramulator2/
cp "${PATCH_SRC}/HBM3_base.yaml" ramulator2/
cp "${PATCH_SRC}/hbm3_linear_mappers.cpp" ramulator2/src/addr_mapper/impl/
cp "${PATCH_SRC}/hbm3_pim_linear_mappers.cpp" ramulator2/src/addr_mapper/impl/
cp "${PATCH_SRC}/HBM3-PIM.cpp" ramulator2/src/dram/impl/
cp "${PATCH_SRC}/hbm3_controller.cpp" ramulator2/src/dram_controller/impl/
cp "${PATCH_SRC}/hbm3_pim_controller.cpp" ramulator2/src/dram_controller/impl/
cp "${PATCH_SRC}/hbm3_trace_recorder.cpp" ramulator2/src/dram_controller/impl/plugin/
cp "${PATCH_SRC}/all_bank_refresh_hbm3.cpp" ramulator2/src/dram_controller/impl/refresh/
cp "${PATCH_SRC}/no_refresh.cpp" ramulator2/src/dram_controller/impl/refresh/
cp "${PATCH_SRC}/pim_scheduler.cpp" ramulator2/src/dram_controller/impl/scheduler/
cp "${PATCH_SRC}/pim_loadstore_trace.cpp" ramulator2/src/frontend/impl/memory_trace/
cp "${PATCH_SRC}/PIM_DRAM_system.cpp" ramulator2/src/memory_system/impl/
cp -r "${PATCH_SRC}/trace_gen" ramulator2/
cp -r "${PATCH_SRC}/patches" ramulator2/

# Apply patches
cd ramulator2;

for f in ./patches/*.patch
do
    patch -p1 < $f
done
