# MIPI
set_property DIFF_TERM_ADV TERM_100 [get_ports {mipi_phy_if_clk_p}]
set_property DIFF_TERM_ADV TERM_100 [get_ports {mipi_phy_if_clk_n}]
set_property DIFF_TERM_ADV TERM_100 [get_ports {mipi_phy_if_data_p[*]}]
set_property DIFF_TERM_ADV TERM_100 [get_ports {mipi_phy_if_data_n[*]}]

# I2C signals --> I2C switch
set_property PACKAGE_PIN G11 [get_ports {iic_scl_io}]
set_property PACKAGE_PIN F10 [get_ports {iic_sda_io}]
set_property IOSTANDARD LVCMOS33 [get_ports {iic_scl_io iic_sda_io}]
set_property SLEW SLOW [get_ports {iic_scl_io iic_sda_io}]
set_property DRIVE 4 [get_ports {iic_scl_io iic_sda_io}]

# Raspi Enable HDA09
# Use only if raspi_enable exists in the wrapper.
set_property PACKAGE_PIN F11 [get_ports {raspi_enable}]
set_property IOSTANDARD LVCMOS33 [get_ports {raspi_enable}]
set_property SLEW SLOW [get_ports {raspi_enable}]
set_property DRIVE 4 [get_ports {raspi_enable}]

set_property BITSTREAM.CONFIG.OVERTEMPSHUTDOWN ENABLE [current_design]