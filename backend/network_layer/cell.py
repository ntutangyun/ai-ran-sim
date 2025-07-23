import math
import settings
from utils import dist_between, estimate_throughput


class Cell:
    def __init__(self, base_station, cell_init_data):
        assert base_station is not None, "Base station cannot be None"
        assert cell_init_data is not None, "Cell init data cannot be None"
        self.base_station = base_station

        self.cell_id = cell_init_data["cell_id"]
        self.frequency_band = cell_init_data["frequency_band"]
        self.carrier_frequency_MHz = cell_init_data["carrier_frequency_MHz"]
        self.bandwidth_Hz = cell_init_data["bandwidth_Hz"]
        self.max_prb = cell_init_data["max_prb"]
        self.max_dl_prb = cell_init_data["max_dl_prb"]
        self.max_ul_prb = cell_init_data["max_ul_prb"]
        self.cell_radius = cell_init_data["cell_radius"]
        self.transmit_power_dBm = cell_init_data["transmit_power_dBm"]
        self.cell_individual_offset_dBm = cell_init_data["cell_individual_offset_dBm"]
        self.frequency_priority = cell_init_data["frequency_priority"]
        self.qrx_level_min = cell_init_data["qrx_level_min"]

        self.prb_ue_allocation_dict = {}  # { "ue_imsi": {"downlink": 30, "uplink": 5}}
        self.connected_ue_list = {}
        self.ue_uplink_signal_strength_dict = {}

    def __repr__(self):
        return f"Cell({self.cell_id}, base_station={self.base_station.bs_id}, frequency_band={self.frequency_band}, carrier_frequency_MHz={self.carrier_frequency_MHz})"

    @property
    def allocated_dl_prb(self):
        return sum(
            [
                self.prb_ue_allocation_dict[ue_imsi]["downlink"]
                for ue_imsi in self.connected_ue_list.keys()
            ]
        )

    @property
    def allocated_ul_prb(self):
        return sum(
            [
                self.prb_ue_allocation_dict[ue_imsi]["uplink"]
                for ue_imsi in self.connected_ue_list.keys()
            ]
        )

    @property
    def allocated_prb(self):
        return sum(
            [
                self.prb_ue_allocation_dict[ue_imsi]["uplink"]
                + self.prb_ue_allocation_dict[ue_imsi]["downlink"]
                for ue_imsi in self.connected_ue_list.keys()
            ]
        )

    @property
    def current_load(self):
        return self.allocated_prb / self.max_prb

    @property
    def current_dl_load(self):
        return self.allocated_dl_prb / self.max_dl_prb

    @property
    def current_ul_load(self):
        return self.allocated_ul_prb / self.max_ul_prb

    @property
    def position_x(self):
        return self.base_station.position_x

    @property
    def position_y(self):
        return self.base_station.position_y

    def register_ue(self, ue):
        self.connected_ue_list[ue.ue_imsi] = ue
        self.prb_ue_allocation_dict[ue.ue_imsi] = {
            "downlink": 0,
            "uplink": 0,
        }

    def monitor_ue_signal_strength(self):
        self.ue_uplink_signal_strength_dict = {}
        pass_loss_model = settings.CHANNEL_PASS_LOSS_MODEL_MAP[
            settings.CHANNEL_PASS_LOSS_MODEL_URBAN_MACRO_NLOS
        ]
        # monitor the ue uplink signal strength
        for ue in self.connected_ue_list.values():
            # calculate the received power based on distance and transmit power
            distance = dist_between(
                self.position_x,
                self.position_y,
                ue.position_x,
                ue.position_y,
            )
            received_power = ue.uplink_transmit_power_dBm - pass_loss_model(
                distance_m=distance, frequency_ghz=self.carrier_frequency_MHz / 1000
            )
            self.ue_uplink_signal_strength_dict[ue.ue_imsi] = received_power

    def select_ue_mcs(self):
        for ue in self.connected_ue_list.values():
            ue.set_downlink_mcs_index(-1)
            ue.set_downlink_mcs_data(None)
            ue_cqi_mcs_data = settings.UE_CQI_MCS_SPECTRAL_EFFICIENCY_TABLE.get(
                ue.downlink_cqi, None
            )
            if ue.downlink_cqi == 0 or ue_cqi_mcs_data is None:
                continue

            ue_cqi_eff = ue_cqi_mcs_data["spectral_efficiency"]
            max_mcs_index = 0
            for (
                mcs_index,
                mcs_eff,
            ) in settings.RAN_MCS_SPECTRAL_EFFICIENCY_TABLE.items():
                if mcs_eff["spectral_efficiency"] <= ue_cqi_eff:
                    max_mcs_index = mcs_index
                else:
                    break
            ue.set_downlink_mcs_index(max_mcs_index)
            downlink_mcs_data = settings.RAN_MCS_SPECTRAL_EFFICIENCY_TABLE.get(
                max_mcs_index, None
            )
            if downlink_mcs_data is None:
                ue.set_downlink_mcs_data(None)
            else:
                # copy the dictionary to avoid modifying the original data
                ue.set_downlink_mcs_data(downlink_mcs_data.copy())

    def step(self, delta_time):
        self.monitor_ue_signal_strength()

        # select modulation and coding scheme (MCS) for each UE based on CQI
        self.select_ue_mcs()

        # allocate PRBs dynamically based on each UE's QoS profile and channel conditions
        self.allocate_prb()

        # for each UE, estimate the downlink, uplink bitrate and latency
        self.estimate_ue_bitrate_and_latency()

    def allocate_prb(self):
        # QoS-aware Proportional Fair Scheduling (PFS)

        # reset PRB allocation for all UEs
        for ue in self.connected_ue_list.values():
            self.prb_ue_allocation_dict[ue.ue_imsi]["downlink"] = 0
            self.prb_ue_allocation_dict[ue.ue_imsi]["uplink"] = 0

        # sample QoS and channel condition-aware PRB allocation
        ue_prb_requirements = {}

        # Step 1: Calculate required PRBs for GBR
        for ue in self.connected_ue_list.values():
            dl_gbr = ue.qos_profile["GBR_DL"]
            dl_mcs = ue.downlink_mcs_data  # Assume this attribute exists
            if dl_mcs is None:
                print(
                    f"Cell {self.cell_id}: UE {ue.ue_imsi} has no downlink MCS data. Skipping."
                )
                continue
            dl_throughput_per_prb = estimate_throughput(
                dl_mcs["modulation_order"], dl_mcs["target_code_rate"], 1
            )
            dl_required_prbs = math.ceil(dl_gbr / dl_throughput_per_prb)
            ue_prb_requirements[ue.ue_imsi] = {
                "dl_required_prbs": dl_required_prbs,
                "dl_throughput_per_prb": dl_throughput_per_prb,
            }

        # Step 2: Allocate PRBs to meet GBR
        dl_total_prb_demand = sum(
            req["dl_required_prbs"] for req in ue_prb_requirements.values()
        )

        if dl_total_prb_demand <= self.max_dl_prb:
            # allocate PRBs based on the required PRBs
            for ue_imsi, req in ue_prb_requirements.items():
                self.prb_ue_allocation_dict[ue_imsi]["downlink"] = req[
                    "dl_required_prbs"
                ]
        else:
            # allocate PRBs based on the proportion
            # first allocate at least one PRB to each UE to ensure minimum service
            dl_remaining_prbs = self.max_dl_prb
            for ue in self.connected_ue_list.values():
                prb = min(1, dl_remaining_prbs)
                self.prb_ue_allocation_dict[ue.ue_imsi]["downlink"] = prb
                dl_remaining_prbs -= prb

            # then allocate the remaining PRBs based on the proportion
            if dl_remaining_prbs > 0:
                for ue_imsi, req in ue_prb_requirements.items():
                    share = req["dl_required_prbs"] / dl_total_prb_demand
                    additional_prbs = int(share * dl_remaining_prbs)
                    self.prb_ue_allocation_dict[ue_imsi]["downlink"] += additional_prbs

        # # Logging
        # for ue_imsi, allocation in self.prb_ue_allocation_dict.items():
        #     print(
        #         f"Cell: {self.cell_id} allocated {allocation['downlink']} DL PRBs for UE {ue_imsi}"
        #     )

    def estimate_ue_bitrate_and_latency(self):
        for ue in self.connected_ue_list.values():
            if ue.downlink_mcs_data is None:
                print(
                    f"Cell {self.cell_id}: UE {ue.ue_imsi} has no downlink MCS data. Skipping."
                )
                continue
            ue_modulation_order = ue.downlink_mcs_data["modulation_order"]
            ue_code_rate = ue.downlink_mcs_data["target_code_rate"]
            ue_dl_prb = self.prb_ue_allocation_dict[ue.ue_imsi]["downlink"]
            # TODO: uplink bitrate
            dl_bitrate = estimate_throughput(
                ue_modulation_order, ue_code_rate, ue_dl_prb
            )
            ue.set_downlink_bitrate(dl_bitrate)
            # TODO: downlink and uplink latency

    def deregister_ue(self, ue):
        if ue.ue_imsi in self.prb_ue_allocation_dict:
            del self.prb_ue_allocation_dict[ue.ue_imsi]
            print(f"Cell {self.cell_id}: Released resources for UE {ue.ue_imsi}")
        else:
            print(f"Cell {self.cell_id}: No resources to release for UE {ue.ue_imsi}")

        if ue.ue_imsi in self.connected_ue_list:
            del self.connected_ue_list[ue.ue_imsi]
            print(f"Cell {self.cell_id}: Deregistered UE {ue.ue_imsi}")
        else:
            print(f"Cell {self.cell_id}: No UE {ue.ue_imsi} to deregister")

    def to_json(self):
        return {
            "cell_id": self.cell_id,
            "frequency_band": self.frequency_band,
            "carrier_frequency_MHz": self.carrier_frequency_MHz,
            "bandwidth_Hz": self.bandwidth_Hz,
            "max_prb": self.max_prb,
            "cell_radius": self.cell_radius,
            "vis_cell_radius": self.cell_radius
            * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "vis_position_x": self.position_x * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "vis_position_y": self.position_y * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "prb_ue_allocation_dict": self.prb_ue_allocation_dict,
            "max_dl_prb": self.max_dl_prb,
            "max_ul_prb": self.max_ul_prb,
            "allocated_dl_prb": self.allocated_dl_prb,
            "allocated_ul_prb": self.allocated_ul_prb,
            "current_dl_load": self.allocated_dl_prb / self.max_dl_prb,
            "current_ul_load": self.allocated_ul_prb / self.max_ul_prb,
            "current_load": self.current_load,
            "connected_ue_list": list(self.connected_ue_list.keys()),
        }
