import inspect
from ..tags import KnowledgeTag
from ..relationships import KnowledgeRelationship
from network_layer.base_station import Cell
from ..knowledge_entry import knowledge_entry

SUPPORTED_CELL_ATTRIBUTES = [
    "cell_id",
    "frequency_band",
    "carrier_frequency_MHz",
    "bandwidth_Hz",
    "max_prb",
    "max_dl_prb",
    "max_ul_prb",
    # "cell_radius",
    "transmit_power_dBm",
    "cell_individual_offset_dBm",
    "frequency_priority",
    "qrx_level_min",
    "prb_ue_allocation_dict",
    "connected_ue_list",
    "allocated_dl_prb",
    "allocated_ul_prb",
    "current_dl_load",
    "current_ul_load",
    "current_load",
    "position_x",
    "position_y",
]

SUPPORTED_CELL_METHODS = [
    "register_ue",
    "monitor_ue_signal_strength",
    "select_ue_mcs",
    "allocate_prb",
    "estimate_ue_bitrate_and_latency",
    "deregister_ue",
    "step",
    # "to_json",
]


@knowledge_entry(
    key="/docs/cells",
    tags=[KnowledgeTag.CELL, KnowledgeTag.KNOWLEDGE_GUIDE],
    related=[],
)
def cell_knowledge_help(sim, knowledge_router, query_key, params):
    return (
        "Welcome to the Cell knowledge base!\n\n"
        "You can query live data and explanations for Cells in the simulation.\n\n"
        "### Available Endpoints:\n"
        "- **List all Cells (identifiers only)**: `/cells`\n"
        "- **Get all attributes values for a specific Cell**: `/cells/{cell_id}`\n"
        "- **Get a specific attribute value of a specific Cell**: `/cells/{cell_id}/attributes/{attribute_name}`\n"
        "- **Explain what an attribute of Cell means**: `/docs/cells/attributes/{attribute_name}`\n"
        "- **Explain what a method in Cell class does**: `/docs/cells/methods/{method_name}`\n"
        "### Supported Cell Attributes:\n"
        f"{', '.join(SUPPORTED_CELL_ATTRIBUTES)}\n\n"
        "### Supported Cell Methods:\n"
        f"{', '.join(SUPPORTED_CELL_METHODS)}\n\n"
    )


# ------------------------------------------
#     GET /cells
#       → List all Cells (representations) in the simulation
# ------------------------------------------
@knowledge_entry(
    key="/cells",
    tags=[KnowledgeTag.CELL],
    related=[],
)
def get_ue_repr_list(sim, knowledge_router, query_key, params):
    list_of_cell_repr = [repr(cell) for cell in sim.cell_list.values()]
    return (
        f"Currently there are {len(list_of_cell_repr)} Cells in the simulation:\n"
        f"{"\n".join(list_of_cell_repr)}\n"
    )


# ------------------------------------------
#     GET /cells/{cell_id}
#       → List all attributes values for the given Cell
# ------------------------------------------
@knowledge_entry(
    key="/cells/{cell_id}",
    tags=[KnowledgeTag.CELL],
    related=[],
)
def get_cell_attributes(sim, knowledge_router, query_key, params):
    cell_id = params["cell_id"]
    cell = sim.cell_list.get(cell_id)
    if cell is None:
        return f"Cell with ID: {cell_id} not found."
    response = f"Attributes of Cell {cell_id}:\n"
    for attr in SUPPORTED_CELL_ATTRIBUTES:
        value = getattr(cell, attr, None)

        if value is not None:
            if attr == "connected_ue_list":
                response += f"- connected_ue_list: {len(value)} UEs\n"
                for ue in value.values():
                    response += f"  - {repr(ue)}\n"
                continue

        response += f"- {attr}: {repr(value)}\n"

    return response


# ------------------------------------------
#     GET /cells/{cell_id}/attributes/{attribute_name}
#       → Get a specific attribute value for the given Cell
# ------------------------------------------
@knowledge_entry(
    key="/cells/{cell_id}/attributes/{attribute_name}",
    tags=[KnowledgeTag.CELL],
    related=[],
)
def get_cell_attribute_value(sim, knowledge_router, query_key, params):
    cell_id = params["cell_id"]
    attribute_name = params["attribute_name"]
    cell = sim.cell_list.get(cell_id)
    if cell is None:
        return f"Cell with ID: {cell_id} not found."
    if attribute_name not in SUPPORTED_CELL_ATTRIBUTES:
        return f"Attribute '{attribute_name}' is not supported."
    value = getattr(cell, attribute_name, None)

    if value is not None:
        if attribute_name == "connected_ue_list":
            return f"Connected UEs: {len(value)}\n" + "\n".join(
                [repr(ue) for ue in value.values()]
            )

    return f"Value of {attribute_name} for Cell {cell_id}: {repr(value)}"


@knowledge_entry(
    "/docs/cells/attributes/cell_id",
    tags=[KnowledgeTag.CELL, KnowledgeTag.ID],
    related=[],
)
def cell_id_explainer(sim, knowledge_router, query_key, params):
    return "The `cell_id` attribute is a unique identifier assigned to each cell in the network. "


@knowledge_entry(
    "/docs/cells/attributes/frequency_band",
    tags=[KnowledgeTag.CELL],
    related=[],
)
def cell_frequency_band_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `frequency_band` attribute specifies the radio frequency band (e.g., n78, n258) in which a cell operates. "
        "This in real-life determines the portion of the radio spectrum allocated to the cell, which in turn affects propagation characteristics, coverage, and capacity. "
        "Different frequency bands may be used for different deployment scenarios, such as urban, suburban, or rural environments."
    )


@knowledge_entry(
    "/docs/cells/attributes/max_dl_prb",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS],
    related=[
        (KnowledgeRelationship.CONSTRAINED_BY, "/docs/cells/attributes/max_prb"),
        (KnowledgeRelationship.USED_BY_METHOD, "/docs/cells/methods/allocate_prb"),
    ],
)
def cell_max_dl_prb_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `max_dl_prb` attribute defines the maximum number of Physical Resource Blocks (PRBs) available for downlink (DL) transmission in a cell. "
        "PRBs are the smallest unit of radio resource allocation in the system. "
        "Downlink PRBs are allocated to user equipments (UEs) for data transmission from the cell to the UE. "
    )


@knowledge_entry(
    "/docs/cells/attributes/max_ul_prb",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS],
    related=[
        (KnowledgeRelationship.CONSTRAINED_BY, "/docs/cells/attributes/max_prb"),
    ],
)
def cell_max_ul_prb_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `max_ul_prb` attribute specifies the maximum number of Physical Resource Blocks (PRBs) available for uplink (UL) transmission in a cell. "
        "Uplink PRBs are used for data sent from the UE to the cell. "
    )


@knowledge_entry(
    "/docs/cells/attributes/transmit_power_dBm",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/user_equipments/methods/monitor_signal_strength",
        )
    ],
)
def cell_transmit_power_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `transmit_power_dBm` attribute specifies the transmit power of a cell, measured in decibel-milliwatts (dBm). "
        "This value directly affects the received signal strength at UEs, influencing coverage area and signal quality. "
        "Higher transmit power can improve coverage but may also increase interference to neighboring cells. "
        "Network operators configure this attribute to balance coverage, capacity, and interference management."
    )


@knowledge_entry(
    "/docs/cells/attributes/cell_individual_offset_dBm",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/user_equipments/methods/monitor_signal_strength",
        )
    ],
)
def cell_individual_offset_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `cell_individual_offset_dBm` attribute is a cell-specific offset (in dB) applied to signal measurements for handover and cell selection decisions. "
        "It allows network operators to bias UE association towards or away from specific cells, supporting load balancing and coverage optimization. "
        "A positive offset makes a cell more attractive for selection, while a negative offset makes it less attractive."
    )


@knowledge_entry(
    "/docs/cells/attributes/frequency_priority",
    tags=[KnowledgeTag.CELL],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/user_equipments/methods/cell_selection_and_camping",
        )
    ],
)
def cell_frequency_priority_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `frequency_priority` attribute indicates the relative priority of a cell's frequency band for UE selection and reselection. "
        "Cells with higher frequency priority values are preferred by UEs when multiple candidate cells are available. "
        "This mechanism helps guide UE distribution across the network and can be used to optimize resource utilization."
    )


@knowledge_entry(
    "/docs/cells/attributes/qrx_level_min",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/user_equipments/methods/monitor_signal_strength",
        )
    ],
)
def cell_qrx_level_min_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `qrx_level_min` attribute defines the minimum required received signal level (in dBm) for a UE to camp on or connect to a cell. "
        "This threshold ensures that UEs only associate with cells where adequate signal quality is available, helping maintain service reliability and user experience."
    )


@knowledge_entry(
    "/docs/cells/attributes/prb_ue_allocation_dict",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS],
    related=[
        (KnowledgeRelationship.SET_BY_METHOD, "/docs/cells/methods/allocate_prb"),
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/cells/methods/estimate_ue_bitrate_and_latency",
        ),
    ],
)
def cell_prb_ue_allocation_dict_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `prb_ue_allocation_dict` attribute is a dictionary mapping each connected UE's identifier to its allocated number of downlink and uplink PRBs. "
        "For example: {'IMSI_1': {'downlink': 10, 'uplink': 5}, ...}. "
        "This structure is updated each scheduling cycle (simulation step) by the cell's resource allocation logic and reflects the current radio resource allocation for all UEs in the cell."
    )


@knowledge_entry(
    "/docs/cells/attributes/allocated_dl_prb",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.DERIVED_FROM,
            "/docs/cells/attributes/prb_ue_allocation_dict",
        ),
    ],
)
def cell_allocated_dl_prb_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `allocated_dl_prb` attribute represents the total number of downlink PRBs currently allocated to all UEs in a cell. "
        "It is computed as the sum of the 'downlink' PRB allocations for each UE in the `prb_ue_allocation_dict`. "
        "This value provides a snapshot of downlink resource usage in the cell at any given time."
    )


@knowledge_entry(
    "/docs/cells/attributes/allocated_ul_prb",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.DERIVED_FROM,
            "/docs/cells/attributes/prb_ue_allocation_dict",
        ),
    ],
)
def cell_allocated_ul_prb_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `allocated_ul_prb` attribute represents the total number of uplink PRBs currently allocated to all UEs in a cell. "
        "It is computed as the sum of the 'uplink' PRB allocations for each UE in the `prb_ue_allocation_dict`. "
        "This value provides a snapshot of uplink resource usage in the cell at any given time."
    )


@knowledge_entry(
    "/docs/cells/attributes/current_dl_load",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.DERIVED_FROM,
            "/docs/cells/attributes/allocated_dl_prb",
        ),
        (
            KnowledgeRelationship.DERIVED_FROM,
            "/docs/cells/attributes/max_dl_prb",
        ),
    ],
)
def cell_current_dl_load_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `current_dl_load` attribute indicates the current downlink load of a cell, expressed as the ratio of allocated downlink PRBs to the maximum available downlink PRBs. "
        "Formula: allocated_dl_prb / max_dl_prb. "
        "This metric reflects how much of the cell's downlink capacity is currently in use and is useful for monitoring congestion and resource utilization."
    )


@knowledge_entry(
    "/docs/cells/attributes/current_ul_load",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.DERIVED_FROM,
            "/docs/cells/attributes/allocated_ul_prb",
        ),
        (
            KnowledgeRelationship.DERIVED_FROM,
            "/docs/cells/attributes/max_ul_prb",
        ),
    ],
)
def cell_current_ul_load_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `current_ul_load` attribute indicates the current uplink load of a cell, expressed as the ratio of allocated uplink PRBs to the maximum available uplink PRBs. "
        "Formula: allocated_ul_prb / max_ul_prb. "
        "This metric reflects how much of the cell's uplink capacity is currently in use and helps identify potential bottlenecks."
    )


@knowledge_entry(
    "/docs/cells/attributes/position_x",
    tags=[KnowledgeTag.CELL, KnowledgeTag.LOCATION],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/cells/methods/monitor_ue_signal_strength",
        ),
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/user_equipments/methods/monitor_signal_strength",
        ),
    ],
)
def cell_position_x_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `position_x` attribute gives the X-coordinate of a cell's location in the simulation environment. "
        "This value is typically inherited from the base station's position and is used for distance calculations, signal strength estimation, and visualization of the network topology."
    )


@knowledge_entry(
    "/docs/cells/attributes/position_y",
    tags=[KnowledgeTag.CELL, KnowledgeTag.LOCATION],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/cells/methods/monitor_ue_signal_strength",
        ),
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/user_equipments/methods/monitor_signal_strength",
        ),
    ],
)
def cell_position_y_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `position_y` attribute gives the Y-coordinate of a cell's location in the simulation environment. "
        "Like `position_x`, this value is used for spatial calculations, signal strength estimation, and visualization."
    )


@knowledge_entry(
    "/docs/cells/attributes/carrier_frequency_MHz",
    tags=[KnowledgeTag.CELL],
    related=[],
)
def cell_carrier_frequency_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `carrier_frequency_MHz` attribute specifies the center frequency (in MHz) at which a cell operates. "
        "This value determines the exact location of the cell's allocated spectrum within its frequency band and is critical for radio planning and interference management."
    )


@knowledge_entry(
    "/docs/cells/attributes/bandwidth_Hz",
    tags=[KnowledgeTag.CELL],
    related=[],
)
def cell_bandwidth_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `bandwidth_Hz` attribute defines the total bandwidth (in Hz) allocated to a cell. "
        "This bandwidth determines the maximum data rate the cell can support and influences how many PRBs are available for allocation to UEs."
    )


@knowledge_entry(
    "/docs/cells/attributes/max_prb",
    tags=[KnowledgeTag.CELL],
    related=[],
)
def cell_max_prb_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `max_prb` attribute specifies the total number of Physical Resource Blocks (PRBs) supported by a cell. "
        "These PRBs are divided between downlink and uplink transmissions and represent the fundamental units of radio resource allocation in the system."
    )


@knowledge_entry(
    "/docs/cells/attributes/connected_ue_list",
    tags=[KnowledgeTag.CELL, KnowledgeTag.UE],
    related=[],
)
def cell_connected_ue_list_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `connected_ue_list` attribute contains a list or dictionary of all UEs currently connected to a cell. "
        "This list is dynamically updated as UEs register, deregister, or handover between cells, and is essential for managing resource allocation and mobility."
    )


@knowledge_entry(
    "/docs/cells/methods/register_ue",
    tags=[KnowledgeTag.CELL, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/cells/attributes/connected_ue_list",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/base_station/method/handle_ue_authentication_and_registration",
        ),
    ],
)
def cell_register_ue_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(Cell, "register_ue"))
    explanation = "The `register_ue` method adds a UE to the cell's list of connected UEs and initializes its PRB allocation (downlink: 0, uplink: 0)."
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/cells/methods/allocate_prb",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/cells/attributes/prb_ue_allocation_dict",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/cells/methods/step",
        ),
    ],
)
def cell_allocate_prb_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(Cell, "allocate_prb"))
    explanation = (
        "The `allocate_prb` method in the Cell class performs QoS-aware Proportional Fair Scheduling (PFS) to allocate Physical Resource Blocks (PRBs) among all connected UEs. "
        "The allocation process is as follows:\n\n"
        "1. **Reset PRB Allocation:** All UEs' downlink and uplink PRB allocations are reset to zero at the start of each allocation cycle.\n\n"
        "2. **Calculate PRB Requirements:** For each connected UE, the method calculates the number of downlink PRBs required to meet its Guaranteed Bit Rate (GBR) using the UE's QoS profile and current downlink MCS (modulation and coding scheme). "
        "The required PRBs are computed by dividing the UE's GBR_DL by the estimated throughput per PRB (which depends on the modulation order and code rate from the MCS).\n\n"
        "3. **Total PRB Demand:** The method sums the required PRBs for all UEs to determine the total downlink PRB demand.\n\n"
        "4. **PRB Allocation:**\n"
        "   - If the total PRB demand is less than or equal to the cell's maximum available downlink PRBs, each UE is allocated the exact number of PRBs it requires.\n"
        "   - If the total demand exceeds the available PRBs, the method first allocates at least one PRB to each UE to ensure minimum service. "
        "The remaining PRBs are then distributed proportionally based on each UE's required share of the total demand.\n\n"
        "This approach ensures that UEs with higher QoS requirements or better channel conditions receive more resources, while still providing a minimum allocation to all UEs. "
        "The final allocation is stored in the `prb_ue_allocation_dict` attribute, which maps each UE's IMSI to its downlink and uplink PRB allocation."
    )

    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/cells/methods/monitor_ue_signal_strength",
    tags=[KnowledgeTag.CELL, KnowledgeTag.CODE, KnowledgeTag.SIMULATION],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/cells/attributes/ue_uplink_signal_strength_dict",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/cells/methods/step",
        ),
    ],
)
def cell_monitor_ue_signal_strength_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(Cell, "monitor_ue_signal_strength"))
    explanation = (
        "The `monitor_ue_signal_strength` method in the Cell class measures and updates the uplink signal strength for each connected UE. "
        "This is a key step in simulating realistic radio conditions and is typically called at every simulation time step.\n\n"
        "The process works as follows:\n"
        "1. The method initializes or clears the cell's `ue_uplink_signal_strength_dict`, which will store the measured uplink signal strength (in dBm) for each UE.\n"
        "2. For each connected UE, it calculates the distance between the cell and the UE using their current positions (`position_x`, `position_y`).\n"
        "3. It retrieves the UE's uplink transmit power (in dBm).\n"
        "4. The method applies a path loss model (as configured in the simulation settings, e.g., Urban Macro NLOS) to estimate the signal attenuation over the distance and frequency.\n"
        "5. The received uplink power at the cell is computed as the UE's transmit power minus the path loss.\n"
        "6. The result is stored in `ue_uplink_signal_strength_dict` under the UE's IMSI.\n\n"
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/cells/methods/select_ue_mcs",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/set_downlink_mcs_index",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/set_downlink_mcs_data",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/cells/methods/step",
        ),
    ],
)
def cell_select_ue_mcs_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(Cell, "select_ue_mcs"))
    explanation = (
        "The `select_ue_mcs` method in the Cell class determines and assigns the most suitable Modulation and Coding Scheme (MCS) for each connected UE based on its current Channel Quality Indicator (CQI).\n\n"
        "The process works as follows:\n"
        "1. For each connected UE, the method resets the UE's `downlink_mcs_index` to -1 and `downlink_mcs_data` to None.\n"
        "2. It retrieves the spectral efficiency for the UE's current CQI from the `UE_CQI_MCS_SPECTRAL_EFFICIENCY_TABLE`.\n"
        "3. If the CQI is 0 or there is no matching entry, the UE is skipped (no MCS assigned).\n"
        "4. Otherwise, the method iterates through the available MCS indices in `RAN_MCS_SPECTRAL_EFFICIENCY_TABLE` and selects the highest MCS index whose spectral efficiency does not exceed the UE's CQI spectral efficiency.\n"
        "5. The selected MCS index and its associated parameters (such as modulation order, target code rate, and spectral efficiency) are assigned to the UE's `downlink_mcs_index` and `downlink_mcs_data` attributes.\n\n"
        "This method ensures that each UE is assigned the most aggressive MCS it can reliably support, maximizing throughput while maintaining link reliability. "
        "The assigned MCS is then used in subsequent resource allocation and throughput estimation steps."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/cells/methods/estimate_ue_bitrate_and_latency",
    tags=[KnowledgeTag.CELL, KnowledgeTag.QoS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_bitrate",
        ),
    ],
)
def cell_estimate_ue_bitrate_and_latency_explainer(
    sim, knowledge_router, query_key, params
):
    code = inspect.getsource(getattr(Cell, "estimate_ue_bitrate_and_latency"))
    explanation = (
        "The `estimate_ue_bitrate_and_latency` method in the Cell class calculates the downlink (and potentially uplink) throughput and latency for each connected UE, "
        "based on the PRB (Physical Resource Block) allocation and the selected Modulation and Coding Scheme (MCS) for each UE.\n\n"
        "The process works as follows:\n"
        "1. For each connected UE, the method first checks if the UE has valid downlink MCS data. If not, the UE is skipped for this estimation cycle.\n"
        "2. For UEs with valid MCS data, the method retrieves the modulation order and target code rate from the UE's `downlink_mcs_data` attribute.\n"
        "3. It then obtains the number of downlink PRBs allocated to the UE from the cell's `prb_ue_allocation_dict`.\n"
        "4. The method calls the `estimate_throughput` utility function, passing the modulation order, code rate, and number of PRBs, to compute the estimated downlink bitrate for the UE.\n"
        "5. The computed bitrate is set on the UE using the `set_downlink_bitrate` method.\n"
        "6. (TODO in code) The method is also intended to estimate downlink and uplink latency, but this is not yet implemented.\n\n"
        "This method ensures that each UE's throughput reflects both its channel quality (via MCS) and its allocated radio resources (PRBs). "
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/cells/methods/deregister_ue",
    tags=[KnowledgeTag.CELL, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/cells/attributes/connected_ue_list",
        ),
    ],
)
def cell_deregister_ue_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(Cell, "deregister_ue"))
    explanation = "The `deregister_ue` method removes a UE from the cell's list of connected UEs and releases its allocated PRBs."
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/cells/methods/step",
    tags=[KnowledgeTag.CELL, KnowledgeTag.SIMULATION, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/cells/methods/monitor_ue_signal_strength",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/cells/methods/select_ue_mcs",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/cells/methods/allocate_prb",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/cells/methods/estimate_ue_bitrate_and_latency",
        ),
    ],
)
def cell_step_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(Cell, "step"))
    explanation = (
        "The `step` method advances the simulation state of the Cell by one time step. "
        "It orchestrates the main per-timestep operations for the cell and its connected UEs. The method performs the following actions in sequence:\n\n"
        "1. **Monitor UE Signal Strength:** Calls `monitor_ue_signal_strength()` to update the uplink signal strength measurements for all connected UEs. "
        "This uses the current positions of the cell and UEs, their transmit powers, and the configured path loss model.\n\n"
        "2. **Select UE MCS:** Calls `select_ue_mcs()` to determine the most suitable Modulation and Coding Scheme (MCS) for each UE based on its Channel Quality Indicator (CQI). "
        "This sets the UE's `downlink_mcs_index` and `downlink_mcs_data` attributes, which are used for throughput estimation and resource allocation.\n\n"
        "3. **Allocate PRBs:** Calls `allocate_prb()` to perform QoS-aware Proportional Fair Scheduling (PFS) and allocate Physical Resource Blocks (PRBs) among all connected UEs. "
        "The allocation is based on each UE's QoS profile (e.g., Guaranteed Bit Rate) and current channel conditions (MCS).\n\n"
        "4. **Estimate UE Throughput and Latency:** Calls `estimate_ue_bitrate_and_latency()` to calculate the downlink bitrate for each UE, "
        "using the allocated PRBs and selected MCS. The computed bitrate is set on the UE. (Note: latency estimation is marked as TODO in the code.)\n\n"
        "By executing these steps in order, the `step` method ensures that the cell's resource allocation, link adaptation, and performance metrics are updated each simulation tick, "
        "reflecting the latest network and radio conditions."
    )
    return f"```python\n{code}\n```\n\n{explanation}"
