import inspect
from ..knowledge_entry import knowledge_entry
from ..tags import KnowledgeTag
from ..relationships import KnowledgeRelationship
from network_layer.base_station import BaseStation

SUPPORTED_BS_ATTRIBUTES = [
    "bs_id",
    "position_x",
    "position_y",
    "cell_list",
    "rrc_measurement_events",
    "ue_registry",
    "ue_rrc_meas_events",
    "ue_rrc_meas_event_handlers",
    "ric_control_actions",
]

SUPPORTED_BS_METHODS = [
    "receive_ue_rrc_meas_events",
    "handle_ue_authentication_and_registration",
    "handle_deregistration_request",
    "to_json",
    "init_rrc_measurement_event_handler",
    "process_ric_control_actions",
    "execute_handover",
    "step",
]


@knowledge_entry(
    key="/docs/base_stations",
    tags=[KnowledgeTag.BS, KnowledgeTag.KNOWLEDGE_GUIDE],
    related=[],
)
def bs_knowledge_help(sim, knowledge_router, query_key, params):
    return (
        "📘 **Welcome to the Base Station (BS) Knowledge Base**\n\n"
        "You can query live data and explanations for Base Station (BSs) in the simulation.\n\n"
        "### Available Endpoints:\n"
        "- **List all BSs (identifiers only)**: `/base_stations`\n"
        "- **Get all attributes values for a specific BS**: `/base_stations/{bs_id}`\n"
        "- **Get a specific attribute value of a specific BS**: `/base_stations/{bs_id}/attributes/{attribute_name}`\n"
        "- **Explain what an attribute of BS means**: `/docs/base_stations/attributes/{attribute_name}`\n"
        "- **Explain what a method in BS class does**: `/docs/base_stations/methods/{method_name}`\n"
        "### Supported BS Attributes:\n"
        f"{', '.join(SUPPORTED_BS_ATTRIBUTES)}\n\n"
        "### Supported BS Methods:\n"
        f"{', '.join(SUPPORTED_BS_METHODS)}\n\n"
    )


# ------------------------------------------
#     GET /base_stations
#       → List all BSs (representations) in the simulation
# ------------------------------------------
@knowledge_entry(
    key="/base_stations",
    tags=[KnowledgeTag.BS],
    related=[],
)
def get_bs_repr_list(sim, knowledge_router, query_key, params):
    list_of_bs_repr = [repr(bs) for bs in sim.base_station_list.values()]
    return (
        f"Currently there are {len(list_of_bs_repr)} BSs in the simulation:\n"
        f"{"\n".join(list_of_bs_repr)}\n"
    )


# ------------------------------------------
#     GET /base_stations/{bs_id}
#       → List all attributes values for the given BS
# ------------------------------------------
@knowledge_entry(
    key="/base_stations/{bs_id}",
    tags=[KnowledgeTag.BS],
    related=[],
)
def get_bs_attributes(sim, knowledge_router, query_key, params):
    bs_id = params["bs_id"]
    bs = sim.base_station_list.get(bs_id, None)
    if bs is None:
        return f"Base Station with ID: {bs_id} not found."
    response = f"Attributes of BS {bs_id}:\n"
    for attr in SUPPORTED_BS_ATTRIBUTES:
        value = getattr(bs, attr, None)

        if value is None:
            response += f"- {attr}: None\n"
            continue

        if (
            attr == "cell_list"
            or attr == "ue_registry"
        ):
            response += f"- {attr}: \n"
            for cell in value.values():
                response += f"  - {repr(cell)}\n"
            continue

        if attr == "ric_control_actions":
            if len(value) == 0:
                response += f"- {attr}: Empty\n"
                continue
            response += f"- {attr} (count: {len(value)}): \n"
            for action in value:
                response += f"  - {repr(action)}\n"
            continue

        if attr == "ue_rrc_meas_events":
            response += f"- {attr}: \n"
            for event in value:
                event_id = event["event_id"]
                triggering_ue = event["triggering_ue"]
                current_cell_id = event["current_cell_id"]
                current_cell_signal_power = event["current_cell_signal_power"]
                best_neighbour_cell_id = event["best_neighbour_cell_id"]
                best_neighbour_cell_signal_power = event[
                    "best_neighbour_cell_signal_power"
                ]
                response += f"  - Event(ID={event_id}, reported_by={triggering_ue.ue_imsi}, ue_current_cell={current_cell_id}, current_cell_signal_power={current_cell_signal_power}dBm, best_neighbor_cell={best_neighbour_cell_id}, best_neighbor_cell_signal_power={best_neighbour_cell_signal_power}dBm)\n"
            continue

        if attr == "ue_rrc_meas_event_handlers":
            response += f"- {attr}: \n"
            for event_id, handler in value.items():
                if hasattr(handler, "__self__") and hasattr(handler, "__name__"):
                    class_name = handler.__self__.__class__.__name__
                    method_name = handler.__name__
                    response += (
                        f"  - Event ID: {event_id}, Handler: /ric/xapps/{class_name}\n"
                    )
                else:
                    response += f"  - Event ID: {event_id}, Handler: {handler}\n"
            continue

        response += f"- {attr}: {repr(value)}\n"

    return response


# ------------------------------------------
#     GET /base_stations/{bs_id}/attributes/{attribute_name}
#       → Get a specific attribute value for the given BS
# ------------------------------------------
@knowledge_entry(
    key="/base_stations/{bs_id}/attributes/{attribute_name}",
    tags=[KnowledgeTag.BS],
    related=[],
)
def get_bs_attribute_value(sim, knowledge_router, query_key, params):
    bs_id = params["bs_id"]
    attribute_name = params["attribute_name"]
    bs = sim.base_station_list.get(bs_id, None)
    if bs is None:
        return f"Base Station with ID: {bs_id} not found."
    if attribute_name not in SUPPORTED_BS_ATTRIBUTES:
        return f"Attribute '{attribute_name}' is not supported."
    response = f"Value of attribute of BS {bs_id}: "
    value = getattr(bs, attribute_name, None)
    if value is None:
        response += f"None\n"
        return response

    if (
        attribute_name == "cell_list"
        or attribute_name == "ue_registry"
        or attribute_name == "ric_control_actions"
    ):
        if len(value) == 0:
            response += f"Empty\n"
            return response

        response += f" (count: {len(value)})\n"
        for item in value.values():
            response += f"  - {repr(item)}\n"
        return response

    if attribute_name == "ue_rrc_meas_events":
        if len(value) == 0:
            response += f"Empty\n"
            return response

        response += f" (count: {len(value)})\n"
        for event in value:
            event_id = event["event_id"]
            triggering_ue = event["triggering_ue"]
            current_cell_id = event["current_cell_id"]
            current_cell_signal_power = event["current_cell_signal_power"]
            best_neighbour_cell_id = event["best_neighbour_cell_id"]
            best_neighbour_cell_signal_power = event["best_neighbour_cell_signal_power"]
            response += f"  - Event(ID={event_id}, reported_by={triggering_ue.ue_imsi}, ue_current_cell={current_cell_id}, current_cell_signal_power={current_cell_signal_power}dBm, best_neighbor_cell={best_neighbour_cell_id}, best_neighbor_cell_signal_power={best_neighbour_cell_signal_power}dBm)\n"
        return response

    if attribute_name == "ue_rrc_meas_event_handlers":
        if len(value) == 0:
            response += f"Empty\n"
            return response
        response += f" (count: {len(value)})\n"
        for event_id, handler in value.items():
            if hasattr(handler, "__self__") and hasattr(handler, "__name__"):
                class_name = handler.__self__.__class__.__name__
                method_name = handler.__name__
                response += (
                    f"  - Event ID: {event_id}, Handler: /ric/xapps/{class_name}\n"
                )
            else:
                response += f"  - Event ID: {event_id}, Handler: {handler}\n"
        return response

    response += f"{repr(value)}\n"

    return response


# Attribute explainers


@knowledge_entry(
    "/docs/base_stations/attributes/bs_id",
    tags=[KnowledgeTag.BS, KnowledgeTag.ID],
    related=[],
)
def bs_id_explainer(sim, knowledge_router, query_key, params):
    return "The `bs_id` attribute is a unique identifier assigned to each Base Station (gNB) in the network."


@knowledge_entry(
    "/docs/base_stations/attributes/position_x",
    tags=[KnowledgeTag.BS, KnowledgeTag.LOCATION],
    related=[
        (KnowledgeRelationship.ASSOCIATED_WITH, "/docs/cells/attributes/position_x"),
    ],
)
def bs_position_x_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `position_x` attribute specifies the X-coordinate of the Base Station's location in the simulation environment. "
        "This coordinate, together with `position_y`, determines the physical placement of the base station and its cells. "
    )


@knowledge_entry(
    "/docs/base_stations/attributes/position_y",
    tags=[KnowledgeTag.BS, KnowledgeTag.LOCATION],
    related=[
        (KnowledgeRelationship.ASSOCIATED_WITH, "/docs/cells/attributes/position_y"),
    ],
)
def bs_position_y_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `position_y` attribute specifies the Y-coordinate of the Base Station's location in the simulation environment. "
        "This coordinate, together with `position_x`, determines the physical placement of the base station and its cells."
    )


@knowledge_entry(
    "/docs/base_stations/attributes/cell_list",
    tags=[KnowledgeTag.BS, KnowledgeTag.CELL],
    related=[],
)
def bs_cell_list_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `cell_list` attribute is a dictionary mapping cell IDs to Cell objects managed by this Base Station. "
        "Each Base Station may manage one or more cells, each with its own configuration."
    )


@knowledge_entry(
    "/docs/base_stations/attributes/ue_registry",
    tags=[KnowledgeTag.BS, KnowledgeTag.UE],
    related=[
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/base_stations/methods/handle_ue_authentication_and_registration",
        ),
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/base_stations/methods/handle_deregistration_request",
        ),
    ],
)
def bs_ue_registry_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `ue_registry` attribute is a dictionary mapping UE IMSIs to registration data. "
        "It tracks all UEs currently registered with this Base Station, including their slice type, QoS profile, and serving cell. "
        "This registry is updated during UE authentication, registration, and deregistration, and is essential for managing network state, resource allocation, and mobility."
    )


@knowledge_entry(
    "/docs/base_stations/attributes/rrc_measurement_events",
    tags=[KnowledgeTag.BS, KnowledgeTag.SIMULATION],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/base_stations/methods/init_rrc_measurement_event_handler",
        ),
        (
            KnowledgeRelationship.AFFECTS,
            "/docs/base_stations/attributes/ue_rrc_meas_event_handers",
        ),
    ],
)
def bs_rrc_measurement_events_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `rrc_measurement_events` attribute lists the RRC measurement event types supported by this Base Station. "
        "These events are used for mobility management and handover decisions."
    )


@knowledge_entry(
    "/docs/base_stations/attributes/ue_rrc_meas_events",
    tags=[KnowledgeTag.BS, KnowledgeTag.SIMULATION],
    related=[
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/base_stations/methods/receive_ue_rrc_meas_events",
        )
    ],
)
def bs_ue_rrc_meas_events_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `ue_rrc_meas_events` attribute is a list that stores RRC measurement events reported by UEs to this Base Station. "
        "These events are queued for processing and may trigger actions such as handovers or other mobility management procedures."
    )


@knowledge_entry(
    "/docs/base_stations/attributes/ue_rrc_meas_event_handers",
    tags=[KnowledgeTag.BS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/base_stations/methods/init_rrc_measurement_event_handler",
        ),
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/base_stations/methods/step",
        ),
    ],
)
def bs_ue_rrc_meas_event_handers_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `ue_rrc_meas_event_handers` attribute is a dictionary mapping RRC measurement event IDs to their corresponding handler functions. "
        "Each handler processes a specific type of RRC measurement event when it is received by the Base Station."
    )


@knowledge_entry(
    "/docs/base_stations/attributes/ric_control_actions",
    tags=[KnowledgeTag.BS, KnowledgeTag.SIMULATION],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/base_stations/methods/process_ric_control_actions",
        ),
        (KnowledgeRelationship.SET_BY_METHOD, "/docs/base_stations/methods/step"),
    ],
)
def bs_ric_control_actions_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `ric_control_actions` attribute contains a list of RIC (RAN Intelligent Controller) control actions to be processed by the Base Station. "
        "These actions may include handover commands and other network control operations."
    )


# Method explainers


@knowledge_entry(
    "/docs/base_stations/methods/receive_ue_rrc_meas_events",
    tags=[KnowledgeTag.BS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/user_equipments/methods/check_rrc_meas_events_to_monitor",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/base_stations/attributes/ue_rrc_meas_events",
        ),
    ],
)
def bs_receive_ue_rrc_meas_events_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(BaseStation, "receive_ue_rrc_meas_events"))
    explanation = (
        "The `receive_ue_rrc_meas_events` method is called when a UE reports an RRC measurement event to the Base Station. "
        "It performs several checks to ensure the event is valid (such as verifying the UE and current cell), then appends the event to the Base Station's internal event queue (`ue_rrc_meas_events`). "
        "This queue is processed during each simulation step, allowing the Base Station to react to UE mobility and signal quality changes. "
        "This method is a key entry point for mobility management and is typically triggered by UEs as they monitor their radio environment."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/base_stations/methods/handle_ue_authentication_and_registration",
    tags=[KnowledgeTag.BS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/base_stations/attributes/ue_registry",
        ),
        (KnowledgeRelationship.CALL_METHOD, "/docs/cells/methods/register_ue"),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/user_equipments/methods/authenticate_and_register",
        ),
    ],
)
def bs_handle_ue_auth_and_reg_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(
        getattr(BaseStation, "handle_ue_authentication_and_registration")
    )
    explanation = (
        "The `handle_ue_authentication_and_registration` method authenticates a UE and registers it with the Base Station and its serving cell. "
        "It first interacts with the core network to perform authentication and retrieve the appropriate slice type and QoS profile for the UE. "
        "The method then updates the `ue_registry` to track the UE's registration state and calls the cell's `register_ue` method to allocate radio resources. "
        "This process is essential for enabling UEs to access network services and is a prerequisite for all subsequent communication and mobility events."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/base_stations/methods/handle_deregistration_request",
    tags=[KnowledgeTag.BS, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.CALL_METHOD, "/docs/cells/methods/deregister_ue"),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/base_stations/attributes/ue_registry",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/user_equipments/methods/deregister",
        ),
    ],
)
def bs_handle_deregistration_request_explainer(
    sim, knowledge_router, query_key, params
):
    code = inspect.getsource(getattr(BaseStation, "handle_deregistration_request"))
    explanation = (
        "The `handle_deregistration_request` method processes a UE's deregistration request. "
        "It interacts with the core network to update the UE's state, releases radio resources in the serving cell, and removes the UE from the Base Station's registry. "
        "Additionally, it cleans up any pending RRC measurement events associated with the UE. "
        "This method ensures that network resources are efficiently managed and that UEs are properly removed from the simulation when their session ends."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/base_stations/methods/init_rrc_measurement_event_handler",
    tags=[KnowledgeTag.BS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/base_stations/attributes/ue_rrc_meas_event_handers",
        )
    ],
)
def bs_init_rrc_meas_event_handler_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(BaseStation, "init_rrc_measurement_event_handler"))
    explanation = (
        "The `init_rrc_measurement_event_handler` method registers a handler function for a specific RRC measurement event type. "
        "This allows the Base Station to define custom logic for processing different types of measurement events reported by UEs. "
        "Handlers are stored in the `ue_rrc_meas_event_handers` dictionary and invoked during each simulation step as events are processed. "
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/base_stations/methods/process_ric_control_actions",
    tags=[KnowledgeTag.BS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/base_stations/methods/execute_handover",
        ),
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/base_stations/attributes/ric_control_actions",
        ),
        (KnowledgeRelationship.CALLED_BY_METHOD, "/docs/base_stations/methods/step"),
    ],
)
def bs_process_ric_control_actions_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(BaseStation, "process_ric_control_actions"))
    explanation = (
        "The `process_ric_control_actions` method processes all pending RIC (RAN Intelligent Controller) control actions for the Base Station. "
        "Currently, it primarily handles handover actions, ensuring that only one handover is executed per simulation step. "
        "This is because once a handover for one UE is completed, everything else (load of relevant cells, other UE's SINR etc) is updated. "
        "The method checks for conflicting or duplicate actions, merges or rejects them as needed, and then calls `execute_handover` to perform the actual handover. "
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/base_stations/methods/execute_handover",
    tags=[KnowledgeTag.BS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/base_stations/methods/process_ric_control_actions",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/base_stations/attributes/ue_registry",
        ),
        (KnowledgeRelationship.CALL_METHOD, "/docs/cells/methods/deregister_ue"),
        (KnowledgeRelationship.CALL_METHOD, "/docs/cells/methods/register_ue"),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/execute_handover",
        ),
    ],
)
def bs_execute_handover_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(BaseStation, "execute_handover"))
    explanation = (
        "The `execute_handover` method performs a handover for a UE from a source cell to a target cell, updating all relevant data structures. "
        "It ensures that the UE is properly deregistered from the source cell, registered with the target cell, and that the Base Station's UE registry reflects the new association. "
        "If the handover is inter-base-station, it also updates the registries of both the source and target base stations. "
        "This method is central to mobility management and enables realistic simulation of handover procedures in the RAN."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/base_stations/methods/step",
    tags=[KnowledgeTag.BS, KnowledgeTag.SIMULATION, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.CALL_METHOD, "/docs/cells/methods/step"),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/base_stations/methods/process_ric_control_actions",
        ),
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/base_stations/attributes/ue_rrc_meas_events",
        ),
    ],
)
def bs_step_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(BaseStation, "step"))
    explanation = (
        "The `step` method advances the simulation state of the Base Station by one time step. "
        "It updates all managed cells, processes RRC measurement events reported by UEs, and executes any pending RIC control actions (such as handovers). "
        "This method is called once per simulation tick and is responsible for orchestrating the dynamic behavior of the base station and its associated cells. "
        "It ensures that mobility, resource allocation, and control logic are applied consistently throughout the simulation."
    )
    return f"```python\n{code}\n```\n\n{explanation}"
