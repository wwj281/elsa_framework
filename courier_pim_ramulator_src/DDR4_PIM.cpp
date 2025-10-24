#include "dram/dram.h"
#include "dram/lambdas.h"


namespace Ramulator {

class DDR4PIM : public IDRAM, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, DDR4PIM, "DDR4-PIM", "DDR4-PIM Device Model")

  public:

    inline static const std::map<std::string, Organization> org_presets = {
      // DQ for Pseudo Channel
      // 1/2/3/4R means 1/2/3/4 ranks for 4/8/12/16-Hi stack
      // We refer to JEDEC Standard (JESD238A).
      //   name          density  DQ  Ch Dimm Ra Bg  Ba   Ro     Co
      {"DDR4_32Gb_x16", {32<<10,  16, {1, 1, 2, 8, 4, 1<<14, 1<<11}}},
      {"DDR4_64Gb_x16", {64<<10,  16, {1, 2, 2, 8, 4, 1<<14, 1<<11}}},
    };

    inline static const std::map<std::string, std::vector<int>> timing_presets = {
      //   name       rate   nBL  nCL  nRCD  nRP   nRAS  nRC   nWR  nRTP nCWL nCCDS nCCDL nRRDS nRRDL nWTRS nWTRL nFAW  nRFC nREFI nCS,  tCK_ps
      {"DDR4_3200W",  {3200,   4,  20,  20,   20,   52,   72,   24,   12,  16,   4,    8,   -1,   -1,    4,    12,  -1,  -1,  -1,   2,    625} },
      {"DDR4_3200AA", {3200,   4,  22,  22,   22,   52,   74,   24,   12,  16,   4,    8,   -1,   -1,    4,    12,  -1,  -1,  -1,   2,    625} },
      {"DDR4_3200AC", {3200,   4,  24,  24,   24,   52,   76,   24,   12,  16,   4,    8,   -1,   -1,    4,    12,  -1,  -1,  -1,   2,    625} },
    };


    /************************************************
    *                Organization
    ***********************************************/   
    const int m_internal_prefetch_size = 8;

    inline static constexpr ImplDef m_levels = {
      "channel", "dimm", "rank", "bankgroup", "bank", "row", "column",    
    };


  /************************************************
   *             Requests & Commands
   ***********************************************/
    inline static constexpr ImplDef m_commands = {
      // DRAM commands
      "ACT", 
      "PRE", "PREA", "PRESB", "PREPB",
      "RD",  "WR",  "RDA",  "WRA",
      "REFab", 
      // PIM commands
      "ACTAB", "ACTSB", "ACTPB",
      "MACAB", "MACSB", "MACPB",
      "WRGB", "MVSB", "MVGB", "SFM",
      "SETM", "SETH", "ACC", "AF", "EWMUL", "BARRIER"
    };

    inline static const ImplLUT m_command_scopes = LUT (
      m_commands, m_levels, {
        // DRAM commadns
        {"ACT",   "row"},
        {"PRE",   "bank"},   {"PREA",   "rank"}, {"PRESB", "bank"}, {"PREPB", "bank"},
        {"RD",    "column"}, {"WR",     "column"}, {"RDA",   "column"}, {"WRA",   "column"},
        {"REFab", "rank"},  
        // PIM commadns
        {"ACTAB", "row"},     {"ACTSB", "row"},     {"ACTPB", "row"},
        {"MACAB",  "column"}, {"MACSB",  "column"}, {"MACPB", "column"}, // ACTPB and MACPB are broadcasted to pCHs in a channel
        {"WRGB",  "dimm"},
        {"MVSB",  "bank"},    {"MVGB", "bank"},
        {"SFM",   "channel"},
        {"SETM",  "bank"},    {"SETH", "channel"},
        {"ACC", "dimm"}, {"AF", "dimm"}, {"EWMUL", "dimm"}
      }
    );

    inline static const ImplLUT m_command_meta = LUT<DRAMCommandMeta> (
      m_commands, {
        //            open?   close?   access?  refresh?
        // DRAM commadns
        {"ACT",       {true,   false,   false,   false}},
        {"PRE",       {false,  true,    false,   false}},
        {"PREA",      {false,  true,    false,   false}},
        {"PRESB",     {false,  true,    false,   false}},
        {"PREPB",     {false,  true,    false,   false}},
        {"RD",        {false,  false,   true,    false}},
        {"WR",        {false,  false,   true,    false}},
        {"RDA",       {false,  true,    true,    false}},
        {"WRA",       {false,  true,    true,    false}},
        {"REFab",     {false,  false,   false,   true }},
        // PIM commadns
        {"ACTAB",     {true,   false,   false,   false}},
        {"ACTSB",     {true,   false,   false,   false}},
        {"ACTPB",     {true,   false,   false,   false}},
        {"MACAB",     {false,  false,   true,    false}},
        {"MACSB",     {false,  false,   true,    false}},
        {"MACPB",     {false,  false,   true,    false}},
        {"WRGB",      {false,  false,   false,   false}},
        {"MVSB",      {false,  false,   false,   false}},
        {"MVGB",      {false,  false,   false,   false}},
        {"SFM",       {false,  false,   false,   false}},
        {"SETM",      {false,  false,   false,   false}},
        {"SETH",      {false,  false,   false,   false}},
        {"ACC",       {false,  false,   false,   false}},
        {"AF",        {false,  false,   false,   false}},
        {"EWMUL",     {false,  false,   false,   false}},
        {"BARRIER",   {false,  false,   false,   false}}, // 源代码里没有加这个，是不是漏掉了？
      }
    );

    inline static constexpr ImplDef m_requests = {
      // DRAM requests
      "read", "write", "all-bank-refresh",
      // PIM requests
      "pim-mac-all-bank", "pim-mac-same-bank", "pim-mac-per-bank",
      "pim-write-to-gemv-buffer", "pim-move-to-softmax-buffer", "pim-move-to-gemv-buffer",
      "pim-softmax", "pim-set-model", "pim-set-head", "pim-barrier",
      "pim-accumulate", "pim-activation-function", "pim-elementwise-multiply"
    };

    inline static const ImplLUT m_request_translations = LUT (
      m_requests, m_commands, {
        // DRAM requests
        {"read", "RD"}, {"write", "WR"}, {"all-bank-refresh", "REFab"},
        // PIM requests
        {"pim-mac-all-bank", "MACAB"}, {"pim-mac-same-bank", "MACSB"}, {"pim-mac-per-bank", "MACPB"},
        {"pim-write-to-gemv-buffer", "WRGB"}, {"pim-move-to-softmax-buffer", "MVSB"}, {"pim-move-to-gemv-buffer", "MVGB"},
        {"pim-softmax", "SFM"}, {"pim-set-model", "SETM"}, {"pim-set-head", "SETH"}, {"pim-barrier", "BARRIER"},
        {"pim-accumulate", "ACC"}, {"pim-activation-function", "AF"}, {"pim-elementwise-multiply", "EWMUL"}
      }
    );

   /************************************************
   *                   Timing
   ***********************************************/
    inline static constexpr ImplDef m_timings = {
      "rate", 
      "nBL", "nCL", "nRCD", "nRP", "nRAS", "nRC", "nWR", "nRTP", "nCWL",
      "nCCDS", "nCCDL",
      "nRRDS", "nRRDL",
      "nWTRS", "nWTRL",
      "nFAW",
      "nRFC","nREFI",
      "nCS",
      "tCK_ps"
    };


  /************************************************
   *                 Node States
   ***********************************************/
    inline static constexpr ImplDef m_states = {
      "Opened", "Closed", "PowerUp", "N/A", "Refreshing"
    };

    inline static const ImplLUT m_init_states = LUT (
      m_levels, m_states, {
        {"channel",   "N/A"}, 
        {"dimm",   "N/A"}, 
        {"rank",      "PowerUp"},
        {"bankgroup", "N/A"},
        {"bank",      "Closed"},
        {"row",       "Closed"},
        {"column",    "N/A"},
      }
    );

  public:
    struct Node : public DRAMNodeBase<DDR4PIM> {
      Node(DDR4PIM* dram, Node* parent, int level, int id) : DRAMNodeBase<DDR4PIM>(dram, parent, level, id) {};
    };
    std::vector<Node*> m_channels;
    
    FuncMatrix<ActionFunc_t<Node>>  m_actions;
    FuncMatrix<PreqFunc_t<Node>>    m_preqs;
    FuncMatrix<RowhitFunc_t<Node>>  m_rowhits;
    FuncMatrix<RowopenFunc_t<Node>> m_rowopens;


  public:
    void tick() override {
      m_clk++;
    };

    void init() override {
      RAMULATOR_DECLARE_SPECS();
      set_organization();
      set_timing_vals();

      set_actions();
      set_preqs();
      set_rowhits();
      set_rowopens();
      
      create_nodes();
    };

    void issue_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      m_channels[channel_id]->update_timing(command, addr_vec, m_clk);
      m_channels[channel_id]->update_states(command, addr_vec, m_clk);
    };


    int get_preq_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->get_preq_command(command, addr_vec, m_clk);
    };

    bool check_ready(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_ready(command, addr_vec, m_clk);
    };

    bool check_rowbuffer_hit(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_rowbuffer_hit(command, addr_vec, m_clk);
    };

  private:
    void set_organization() {
      // Channel width
      m_channel_width = param_group("org").param<int>("channel_width").default_val(128);

      // Organization
      m_organization.count.resize(m_levels.size(), -1);

      // Load organization preset if provided
      if (auto preset_name = param_group("org").param<std::string>("preset").optional()) {
        if (org_presets.count(*preset_name) > 0) {
          m_organization = org_presets.at(*preset_name);
        } else {
          throw ConfigurationError("Unrecognized organization preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Override the preset with any provided settings
      if (auto dq = param_group("org").param<int>("dq").optional()) {
        m_organization.dq = *dq;
      }

      for (int i = 0; i < m_levels.size(); i++){
        auto level_name = m_levels(i);
        if (auto sz = param_group("org").param<int>(level_name).optional()) {
          m_organization.count[i] = *sz;
        }
      }

      if (auto density = param_group("org").param<int>("density").optional()) {
        m_organization.density = *density;
      }

      // Sanity check: is the calculated channel density the same as the provided one?
      size_t _density = size_t(m_organization.count[m_levels["bankgroup"]]) *
                        size_t(m_organization.count[m_levels["bank"]]) *
                        size_t(m_organization.count[m_levels["row"]]) *
                        size_t(m_organization.count[m_levels["column"]]) *
                        size_t(m_organization.dq) *
                        size_t(m_internal_prefetch_size);
      _density >>= 20;
      if (m_organization.density != _density) {
        throw ConfigurationError(
            "Calculated {} channel density {} Mb does not equal the provided density {} Mb!", 
            get_name(),
            _density, 
            m_organization.density
        );
      }

    };

    void set_timing_vals() {
      m_timing_vals.resize(m_timings.size(), -1);

      // Load timing preset if provided
      bool preset_provided = false;
      if (auto preset_name = param_group("timing").param<std::string>("preset").optional()) {
        if (timing_presets.count(*preset_name) > 0) {
          m_timing_vals = timing_presets.at(*preset_name);
          preset_provided = true;
        } else {
          throw ConfigurationError("Unrecognized timing preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Check for rate (in MT/s), and if provided, calculate and set tCK (in picosecond)
      if (auto dq = param_group("timing").param<int>("rate").optional()) {
        if (preset_provided) {
          throw ConfigurationError("Cannot change the transfer rate of {} when using a speed preset !", get_name());
        }
        m_timing_vals("rate") = *dq;
      }
      int tCK_ps = 1E6 / (m_timing_vals("rate") / 2); // QDR DQ pins
      m_timing_vals("tCK_ps") = tCK_ps;

      // Load the organization specific timings
      int dq_id = [](int dq) -> int {
        switch (dq) {
          case 4:  return 0;
          case 8:  return 1;
          case 16: return 2;
          default: return -1;
        }
      }(m_organization.dq);

      int rate_id = [](int rate) -> int {
        switch (rate) {
          case 1600:  return 0;
          case 1866:  return 1;
          case 2133:  return 2;
          case 2400:  return 3;
          case 2666:  return 4;
          case 2933:  return 5;
          case 3200:  return 6;
          default:    return -1;
        }
      }(m_timing_vals("rate"));

      // Tables for secondary timings determined by the frequency, density, and DQ width.
      // Defined in the JEDEC standard (e.g., Table 169-170, JESD79-4C).
      constexpr int nRRDS_TABLE[3][7] = {
      // 1600  1866  2133  2400  2666  2933  3200
        { 4,    4,    4,    4,    4,    4,    4},   // x4
        { 4,    4,    4,    4,    4,    4,    4},   // x8
        { 5,    5,    6,    7,    8,    8,    9},   // x16
      };
      constexpr int nRRDL_TABLE[3][7] = {
      // 1600  1866  2133  2400  2666  2933  3200
        { 5,    5,    6,    6,    7,    8,    8 },  // x4
        { 5,    5,    6,    6,    7,    8,    8 },  // x8
        { 6,    6,    7,    8,    9,    10,   11},  // x16
      };
      constexpr int nFAW_TABLE[3][7] = {
      // 1600  1866  2133  2400  2666  2933  3200
        { 16,   16,   16,   16,   16,   16,   16},  // x4
        { 20,   22,   23,   26,   28,   31,   34},  // x8
        { 28,   28,   32,   36,   40,   44,   48},  // x16
      };

      if (dq_id != -1 && rate_id != -1) {
        m_timing_vals("nRRDS") = nRRDS_TABLE[dq_id][rate_id];
        m_timing_vals("nRRDL") = nRRDL_TABLE[dq_id][rate_id];
        m_timing_vals("nFAW")  = nFAW_TABLE [dq_id][rate_id];
      }

      // Refresh timings
      // tRFC table (unit is nanosecond!)
      constexpr int tRFC_TABLE[3][4] = {
      //  2Gb   4Gb   8Gb  16Gb
        { 160,  260,  360,  550}, // Normal refresh (tRFC1)
        { 110,  160,  260,  350}, // FGR 2x (tRFC2)
        { 90,   110,  160,  260}, // FGR 4x (tRFC4)
      };

      // tREFI(base) table (unit is nanosecond!)
      constexpr int tREFI_BASE = 7800;
      int density_id = [](int density_Mb) -> int { 
        switch (density_Mb) {
          case 2048:  return 0;
          case 4096:  return 1;
          case 8192:  return 2;
          case 16384: return 3;
          default:    return -1;
        }
      }(m_organization.density);

      m_timing_vals("nRFC")  = JEDEC_rounding(tRFC_TABLE[0][density_id], tCK_ps);
      m_timing_vals("nREFI") = JEDEC_rounding(tREFI_BASE, tCK_ps);

      // Overwrite timing parameters with any user-provided value
      // Rate and tCK should not be overwritten
      for (int i = 1; i < m_timings.size() - 1; i++) {
        auto timing_name = std::string(m_timings(i));

        if (auto provided_timing = param_group("timing").param<int>(timing_name).optional()) {
          // Check if the user specifies in the number of cycles (e.g., nRCD)
          m_timing_vals(i) = *provided_timing;
        } else if (auto provided_timing = param_group("timing").param<float>(timing_name.replace(0, 1, "t")).optional()) {
          // Check if the user specifies in nanoseconds (e.g., tRCD)
          m_timing_vals(i) = JEDEC_rounding(*provided_timing, tCK_ps);
        }
      }

      // Check if there is any uninitialized timings
      for (int i = 0; i < m_timing_vals.size(); i++) {
        if (m_timing_vals(i) == -1) {
          throw ConfigurationError("In \"{}\", timing {} is not specified!", get_name(), m_timings(i));
        }
      }      

      // Set read latency
      m_read_latency = m_timing_vals("nCL") + m_timing_vals("nBL");

      // Populate the timing constraints
      #define V(timing) (m_timing_vals(timing))
      populate_timingcons(this, {


          /////////////////////////////////
          ////--         PIM           --//
          /////////////////////////////////

          /*** PIM-MAC-All-Bank ***/ 
          /// 2-cycle ACT command (for row commands)
          {.level = "channel", .preceding = {"ACTAB"}, .following = {"ACTAB", "ACT", "PRE", "PREA", "REFab"}, .latency = 2},
          /// All banks in a dimm 
          {.level = "dimm", .preceding = {"MACAB"}, .following = {"MACAB"}, .latency = V("nCCDL")},          
          //{.level = "dimm", .preceding = {"ACTAB"}, .following = {"ACTAB"}, .latency = V("nRC")},
          {.level = "dimm", .preceding = {"ACTAB"}, .following = {"ACTAB"}, .latency = V("nFAW") * size_t(m_organization.count[m_levels["bankgroup"]]) * size_t(m_organization.count[m_levels["bank"]]) / (4 * size_t(m_organization.count[m_levels["rank"]]))},  
          {.level = "dimm", .preceding = {"ACTAB"}, .following = {"MACAB"}, .latency = V("nRCD")},  
          {.level = "dimm", .preceding = {"ACTAB"}, .following = {"PREA"}, .latency = V("nRAS")},  
          {.level = "dimm", .preceding = {"MACAB"},  .following = {"PREA"}, .latency = V("nRTP")},  
          {.level = "dimm", .preceding = {"PREA"}, .following = {"ACTAB"}, .latency = V("nRP")},  
          /// RAS <-> REF
          {.level = "rank", .preceding = {"ACTAB"}, .following = {"REFab"}, .latency = V("nRC")},          
          {.level = "rank", .preceding = {"PREA"}, .following = {"REFab"}, .latency = V("nRP")},          
          {.level = "rank", .preceding = {"REFab"}, .following = {"ACTAB"}, .latency = V("nRFC")},          


          /*** PIM-MAC-Same-Bank ***/ 
          /// 2-cycle ACT command (for row commands)
          {.level = "channel", .preceding = {"ACTSB"}, .following = {"ACTSB", "ACT", "PRE", "PREA", "PRESB", "REFab"}, .latency = 2},
          /// Same-bank MAC timings. The timings of the bank in other BGs will be updated by action function
          {.level = "channel", .preceding = {"MACSB"}, .following = {"MACSB"}, .latency = V("nCCDL")},          
          {.level = "bank", .preceding = {"ACTSB"}, .following = {"ACTSB"}, .latency = V("nRC")},  
          {.level = "bank", .preceding = {"ACTSB"}, .following = {"MACSB"}, .latency = V("nRCD")},
          {.level = "bank", .preceding = {"ACTSB"}, .following = {"PRESB"}, .latency = V("nRAS")},  
          {.level = "bank", .preceding = {"MACSB"},  .following = {"PRESB"}, .latency = V("nRTPL")},  
          {.level = "bank", .preceding = {"PRESB"}, .following = {"ACTSB"}, .latency = V("nRP")},     
          /// RAS <-> REF
          {.level = "rank", .preceding = {"ACTSB"}, .following = {"REFab"}, .latency = V("nRC")},
          {.level = "rank", .preceding = {"PRESB"}, .following = {"REFab"}, .latency = V("nRP")},                    
          {.level = "rank", .preceding = {"REFab"}, .following = {"ACTSB"}, .latency = V("nRFC")},          


          /*** PIM-MAC-Per-Bank ***/      // Broadcasting to pCHs in a channel
          /// 2-cycle ACT command (for row commands)
          {.level = "channel", .preceding = {"ACTPB"}, .following = {"ACTPB", "ACT", "PRE", "PREA", "REFab"}, .latency = 2},
          /// Per-bank MAC timings. The timings of the bank in other pCHs will be updated by action function
          {.level = "channel", .preceding = {"MACPB"}, .following = {"MACPB"}, .latency = V("nBL")},
          {.level = "rank", .preceding = {"MACPB"}, .following = {"MACPB"}, .latency = V("nCCDS")},          
          {.level = "bankgroup", .preceding = {"MACPB"}, .following = {"MACPB"}, .latency = V("nCCDL")},          
          {.level = "bank", .preceding = {"ACTPB"}, .following = {"ACTPB"}, .latency = V("nRC")},  
          {.level = "bank", .preceding = {"ACTPB"}, .following = {"MACPB"}, .latency = V("nRCD")},  
          {.level = "bank", .preceding = {"ACTPB"}, .following = {"PREPB"}, .latency = V("nRAS")},  
          {.level = "bank", .preceding = {"MACPB"},  .following = {"PREPB"}, .latency = V("nRTPL")},  
          {.level = "bank", .preceding = {"PREPB"}, .following = {"ACTPB"}, .latency = V("nRP")},  
          /// RAS <-> REF
          {.level = "rank", .preceding = {"ACTPB"}, .following = {"REFab"}, .latency = V("nRC")},
          {.level = "pseudochannel", .preceding = {"PREPB"}, .following = {"REFab"}, .latency = V("nRP")},                       
          {.level = "rank", .preceding = {"REFab"}, .following = {"ACTPB"}, .latency = V("nRFC")},          


          /*** Data Movement ***/                   // These can be executed simultaneously with MACAB/MACSB/MACPB because their data paths are different from that of MACAB/MACSB/MACPB.
          // CAS <-> CAS (DQ <-> GEMV unit)
          /*** Channel ***/ 
          {.level = "channel", .preceding = {"WRGB"}, .following = {"WRGB"}, .latency = V("nBL")},

          /*** Dimm ***/ 
          {.level = "dimm", .preceding = {"WRGB"}, .following = {"WRGB"}, .latency = V("nCCDL") + V("nCCDS")},
          {.level = "dimm", .preceding = {"WRGB"}, .following = {"ACTAB", "MACAB"}, .latency = V("nCCDL") + V("nCCDS")},
          // {.level = "pseudochannel", .preceding = {"WRGB", "MVSB", "MVGB", "SFM", "RD", "WR"}, .following = {"WRGB", "MVSB", "MVGB", "SFM", "RD", "WR"}, .latency = V("nBL")},
          {.level = "dimm", .preceding = {"MVGB"}, .following = {"ACC", "AF"}, .latency = V("nCCDS") * size_t(m_organization.count[m_levels["rank"]])* size_t(m_organization.count[m_levels["bankgroup"]])  * size_t(m_organization.count[m_levels["bank"]])},
          {.level = "dimm", .preceding = {"ACC"}, .following = {"ACC", "AF"}, .latency = V("nCCDL")},
          {.level = "dimm", .preceding = {"AF"}, .following = {"AF", "EWMUL"}, .latency = V("nCCDL")},
          {.level = "dimm", .preceding = {"EWMUL"}, .following = {"EWMUL", "ACTAB", "MACAB"}, .latency = V("nCCDL")},
          {.level = "dimm", .preceding = {"MVGB"}, .following = {"MVGB"}, .latency = V("nCCDS") * size_t(m_organization.count[m_levels["rank"]]) * size_t(m_organization.count[m_levels["bankgroup"]]) * size_t(m_organization.count[m_levels["bank"]])},
          
          /*** Rank ***/ 
          {.level = "rank", .preceding = {"MVGB"}, .following = {"MACAB"}, .latency = V("nCCDL")},    

          /*** Bank Group ***/ 
          {.level = "bankgroup", .preceding = {"MACAB"}, .following = {"MVGB"}, .latency = V("nCCDL")},         


          /////////////////////////////////
          ////--     DRAM Default      --//
          /////////////////////////////////

          /*** Channel ***/ 
          // CAS <-> CAS
          /// Data bus occupancy
          {.level = "channel", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nBL")},
          {.level = "channel", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nBL")},

          /*** Rank (or different BankGroup) ***/ 
          // CAS <-> CAS
          /// nCCDS is the minimal latency for column commands 
          {.level = "rank", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nCCDS")},
          {.level = "rank", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nCCDS")},
          /// RD <-> WR, Minimum Read to Write, Assuming tWPRE = 1 tCK                          
          {.level = "rank", .preceding = {"RD", "RDA"}, .following = {"WR", "WRA"}, .latency = V("nCL") + V("nBL") + 2 - V("nCWL")},
          /// WR <-> RD, Minimum Read after Write
          {.level = "rank", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCWL") + V("nBL") + V("nWTRS")},
          /// CAS <-> CAS between sibling ranks, nCS (rank switching) is needed for new DQS
          {.level = "rank", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA", "WR", "WRA"}, .latency = V("nBL") + V("nCS"), .is_sibling = true},
          {.level = "rank", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCL")  + V("nBL") + V("nCS") - V("nCWL"), .is_sibling = true},
          /// CAS <-> PREab
          {.level = "rank", .preceding = {"RD"}, .following = {"PREA"}, .latency = V("nRTP")},
          {.level = "rank", .preceding = {"WR"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL") + V("nWR")},          
          /// RAS <-> RAS
          {.level = "rank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDS")},          
          {.level = "rank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nFAW"), .window = 4},          
          {.level = "rank", .preceding = {"ACT"}, .following = {"PREA"}, .latency = V("nRAS")},          
          {.level = "rank", .preceding = {"PREA"}, .following = {"ACT"}, .latency = V("nRP")},          
          /// RAS <-> REF
          {.level = "rank", .preceding = {"ACT"}, .following = {"REFab"}, .latency = V("nRC")},          
          {.level = "rank", .preceding = {"PRE", "PREA"}, .following = {"REFab"}, .latency = V("nRP")},          
          {.level = "rank", .preceding = {"RDA"}, .following = {"REFab"}, .latency = V("nRP") + V("nRTP")},          
          {.level = "rank", .preceding = {"WRA"}, .following = {"REFab"}, .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},          
          {.level = "rank", .preceding = {"REFab"}, .following = {"ACT"}, .latency = V("nRFC")},          

          /*** Same Bank Group ***/ 
          /// CAS <-> CAS
          {.level = "bankgroup", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nCCDL")},          
          {.level = "bankgroup", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nCCDL")},          
          {.level = "bankgroup", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCWL") + V("nBL") + V("nWTRL")},
          /// RAS <-> RAS
          {.level = "bankgroup", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDL")},  

          /*** Bank ***/ 
          {.level = "bank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRC")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"RD", "RDA", "WR", "WRA"}, .latency = V("nRCD")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"PRE"}, .latency = V("nRAS")},  
          {.level = "bank", .preceding = {"PRE"}, .following = {"ACT"}, .latency = V("nRP")},  
          {.level = "bank", .preceding = {"RD"},  .following = {"PRE"}, .latency = V("nRTP")},  
          {.level = "bank", .preceding = {"WR"},  .following = {"PRE"}, .latency = V("nCWL") + V("nBL") + V("nWR")},  
          {.level = "bank", .preceding = {"RDA"}, .following = {"ACT"}, .latency = V("nRTP") + V("nRP")},  
          {.level = "bank", .preceding = {"WRA"}, .following = {"ACT"}, .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")}, 
        }
      );
      #undef V

    };


    // There are no actions and prerequisites for WRGB, MVSB, MVGB, SFM, SETM, SETH because they are not related to the state of the DRAM.

    void set_actions() {
      m_actions.resize(m_levels.size(), std::vector<ActionFunc_t<Node>>(m_commands.size()));

      // Pseudo Channel Actions
      m_actions[m_levels["dimm"]][m_commands["PREA"]] = Lambdas::Action::Channel::PREA<DDR4PIM>;

      // Same-Bank Actions.
      m_actions[m_levels["bank"]][m_commands["PRESB"]] = Lambdas::Action::Bank::PRESB<DDR4PIM>;
      // We call update_timing for the banks in other BGs here
      m_actions[m_levels["bankgroup"]][m_commands["MACSB"]]  = Lambdas::Action::BankGroup::PIMSameBankActions<DDR4PIM>;

      // Per-Bank Actions. (pCH Broadcast)
      m_actions[m_levels["bank"]][m_commands["PREPB"]] = Lambdas::Action::Bank::PREPB<DDR4PIM>;
      // We call update_timing for the bank in other pCH here
      m_actions[m_levels["bankgroup"]][m_commands["MACPB"]]  = Lambdas::Action::BankGroup::PIMPerBankActions<DDR4PIM>;


      // Bank Actions
      m_actions[m_levels["bank"]][m_commands["ACT"]] = Lambdas::Action::Bank::ACT<DDR4PIM>;
      m_actions[m_levels["bank"]][m_commands["PRE"]] = Lambdas::Action::Bank::PRE<DDR4PIM>;
      m_actions[m_levels["bank"]][m_commands["ACTAB"]] = Lambdas::Action::Bank::ACTAB<DDR4PIM>;
      m_actions[m_levels["bank"]][m_commands["ACTSB"]]  = Lambdas::Action::Bank::ACTSB<DDR4PIM>;
      m_actions[m_levels["bank"]][m_commands["ACTPB"]]  = Lambdas::Action::Bank::ACTPB<DDR4PIM>;
      m_actions[m_levels["bank"]][m_commands["RDA"]] = Lambdas::Action::Bank::PRE<DDR4PIM>;
      m_actions[m_levels["bank"]][m_commands["WRA"]] = Lambdas::Action::Bank::PRE<DDR4PIM>;
    };

    void set_preqs() {
      m_preqs.resize(m_levels.size(), std::vector<PreqFunc_t<Node>>(m_commands.size()));

      // Pseudo Channel Preqs
      m_preqs[m_levels["dimm"]][m_commands["REFab"]] = Lambdas::Preq::Channel::RequireAllBanksClosed<DDR4PIM>;

      // Bank Preqs
      m_preqs[m_levels["bank"]][m_commands["RD"]] = Lambdas::Preq::Bank::RequireRowOpen<DDR4PIM>;
      m_preqs[m_levels["bank"]][m_commands["WR"]] = Lambdas::Preq::Bank::RequireRowOpen<DDR4PIM>;
      m_preqs[m_levels["bank"]][m_commands["MACAB"]] = Lambdas::Preq::Bank::RequireAllBanksRowOpen<DDR4PIM>;
      m_preqs[m_levels["bank"]][m_commands["MACSB"]]  = Lambdas::Preq::Bank::RequirePIMSameBanksRowOpen<DDR4PIM>;
      m_preqs[m_levels["bank"]][m_commands["MACPB"]]  = Lambdas::Preq::Bank::RequirePIMPerBanksRowOpen<DDR4PIM>; // pCH Broadcast
    };

    void set_rowhits() {
      m_rowhits.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowhits[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowHit::Bank::RDWR<DDR4PIM>;
      m_rowhits[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowHit::Bank::RDWR<DDR4PIM>;
      m_rowhits[m_levels["bank"]][m_commands["MACAB"]] = Lambdas::RowHit::Bank::RDWR<DDR4PIM>;
      m_rowhits[m_levels["bank"]][m_commands["MACSB"]] = Lambdas::RowHit::Bank::RDWR<DDR4PIM>;
      m_rowhits[m_levels["bank"]][m_commands["MACPB"]] = Lambdas::RowHit::Bank::RDWR<DDR4PIM>;
    }


    void set_rowopens() {
      m_rowopens.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowopens[m_levels["bank"]][m_commands["RD"]] = Lambdas::RowOpen::Bank::RDWR<DDR4PIM>;
      m_rowopens[m_levels["bank"]][m_commands["WR"]] = Lambdas::RowOpen::Bank::RDWR<DDR4PIM>;
      m_rowopens[m_levels["bank"]][m_commands["MACAB"]] = Lambdas::RowOpen::Bank::RDWR<DDR4PIM>;
      m_rowopens[m_levels["bank"]][m_commands["MACSB"]] = Lambdas::RowOpen::Bank::RDWR<DDR4PIM>;
      m_rowopens[m_levels["bank"]][m_commands["MACPB"]] = Lambdas::RowOpen::Bank::RDWR<DDR4PIM>;
    }


    void create_nodes() {
      int num_channels = m_organization.count[m_levels["channel"]];
      for (int i = 0; i < num_channels; i++) {
        Node* channel = new Node(this, nullptr, 0, i);
        m_channels.push_back(channel);
      }
    };
};


}        // namespace Ramulator
