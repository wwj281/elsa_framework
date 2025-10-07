## initialize ramulator2
cd ramulator2
git reset --hard b7c70275f04126c647edb989270cc429776955d1
cd ..

# copy files
cp courier_pim_ramulator_src/DDR4_PIM.cpp ramulator2/src/dram/impl/
cp courier_pim_ramulator_src/ddr_pim_controller.cpp ramulator2/src/dram_controller/impl/
cp courier_pim_ramulator_src/pim_scheduler.cpp ramulator2/src/dram_controller/impl/scheduler/
cp courier_pim_ramulator_src/pim_loadstore_trace.cpp ramulator2/src/frontend/impl/memory_trace/
cp courier_pim_ramulator_src/PIM_DRAM_system.cpp ramulator2/src/memory_system/impl/
cp -r courier_pim_ramulator_src/trace_gen ramulator2/
cp -r courier_pim_ramulator_src/patches1 ramulator2/

# Apply patches
cd ramulator2;

for f in ./patches/*.patch
do
    patch -p1 < $f
done
