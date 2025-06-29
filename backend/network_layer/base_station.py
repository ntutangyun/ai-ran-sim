import time
import settings
from .cell import Cell
from .edge_server import EdgeServer
import logging
import base64



logger = logging.getLogger(__name__)


class BaseStation:
    def __init__(self, simulation_engine, bs_init_data):
        assert simulation_engine is not None, "Simulation engine cannot be None"
        assert bs_init_data is not None, "Base station init data cannot be None"

        self.simulation_engine = simulation_engine
        self.core_network = simulation_engine.core_network

        self.bs_id = bs_init_data["bs_id"]
        self.position_x = bs_init_data["position_x"]
        self.position_y = bs_init_data["position_y"]
        self.cell_list = {}
        for cell_init_data in bs_init_data["cell_list"]:
            new_cell = Cell(
                base_station=self,
                cell_init_data=cell_init_data,
            )
            self.cell_list[cell_init_data["cell_id"]] = new_cell
        self.rrc_measurement_events = bs_init_data["rrc_measurement_events"]

        self.edge_server = EdgeServer(self, bs_init_data["edge_server"])

        self.ue_registry = {}
        self.ue_rrc_meas_events = []
        self.ue_rrc_meas_event_handers = {}

        self.ric_control_actions = []

        self.ai_service_event_handler = None

    def __repr__(self):
        return f"BS {self.bs_id}"

    def receive_ue_rrc_meas_events(self, event):
        # sanity check
        ue = event["triggering_ue"]
        current_cell = self.cell_list.get(event["current_cell_id"], None)
        assert ue is not None, "UE cannot be None"
        assert current_cell is not None, "Current cell cannot be None"
        assert (
            ue.current_cell.cell_id == current_cell.cell_id
        ), f"UE {ue.ue_imsi} (current cell: {ue.current_cell.cell_id}) is not in the current cell ({current_cell.cell_id})"
        logger.info(f"{self} received UE reported RRC measurement event:")
        logger.info(event)
        self.ue_rrc_meas_events.append(event)

    def handle_ue_authentication_and_registration(self, ue):
        core_response = self.core_network.handle_ue_authentication_and_registration(ue)
        ue_reg_data = {
            "ue": ue,
            "slice_type": core_response["slice_type"],
            "qos_profile": core_response["qos_profile"],
            "cell": ue.current_cell,
            "rrc_meas_events": self.rrc_measurement_events.copy(),
        }
        self.ue_registry[ue.ue_imsi] = ue_reg_data
        ue.current_cell.register_ue(ue)
        return ue_reg_data.copy()

    def handle_deregistration_request(self, ue):
        self.core_network.handle_deregistration_request(ue)
        # for simplicity, gNB directly releases resources instead of having AMF to initiate the release
        ue.current_cell.deregister_ue(ue)
        if ue.ue_imsi in self.ue_registry:
            del self.ue_registry[ue.ue_imsi]

        # remove rrc measurement events for the UE
        events_to_remove = []
        for event in self.ue_rrc_meas_events:
            if event["triggering_ue"] == ue:
                events_to_remove.append(event)
        for event in events_to_remove:
            self.ue_rrc_meas_events.remove(event)

        logger.info(
            f"gNB {self.bs_id}: UE {ue.ue_imsi} deregistered and resources released."
        )
        return True

    def to_json(self):
        return {
            "bs_id": self.bs_id,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "vis_position_x": self.position_x * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "vis_position_y": self.position_y * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "ue_registry": list(self.ue_registry.keys()),
            "cell_list": [cell.to_json() for cell in self.cell_list.values()],
            "edge_server": self.edge_server.to_json(),
        }

    def init_rrc_measurement_event_handler(self, event_id, handler):
        assert event_id is not None, "Event ID cannot be None"
        assert handler is not None, "Handler cannot be None"
        assert (
            event_id not in self.ue_rrc_meas_event_handers
        ), f"Handler for event ID {event_id} already registered"
        self.ue_rrc_meas_event_handers[event_id] = handler

    def init_ai_service_event_handler(self, handler):
        assert handler is not None, "Handler cannot be None"
        self.ai_service_event_handler = handler

    def process_ric_control_actions(self):
        # only handover actions are supported for now

        # check if there are multiple handover actions for the same UE,
        # reject or merge wherever necessary
        ue_handover_actions = {}
        for action in self.ric_control_actions:
            if action.action_type != action.ACTION_TYPE_HANDOVER:
                logger.info(
                    f"gNB {self.bs_id}: Ignoring non-handover action: {action.action_type}"
                )
                continue

            ue = action.action_data["ue"]
            if ue.ue_imsi not in ue_handover_actions:
                ue_handover_actions[ue.ue_imsi] = []
            ue_handover_actions[ue.ue_imsi].append(action)

        # process each UE's handover actions
        for ue_imsi, actions in ue_handover_actions.items():
            # for now perform the first handover action only.
            action = actions[0]
            ue = action.action_data["ue"]
            source_cell_id = action.action_data["source_cell_id"]
            target_cell_id = action.action_data["target_cell_id"]
            source_cell = self.simulation_engine.cell_list[source_cell_id]
            target_cell = self.simulation_engine.cell_list[target_cell_id]
            self.execute_handover(ue, source_cell, target_cell)
            break

    def execute_handover(self, ue, source_cell, target_cell):
        assert ue is not None, "UE cannot be None"
        assert (
            source_cell is not None and target_cell is not None
        ), "Source or target cell cannot be None"
        assert source_cell != target_cell, "Source and target cell cannot be the same"
        assert (
            ue.current_cell.cell_id == source_cell.cell_id
        ), f"UE {ue.ue_imsi} (current cell: {ue.current_cell.cell_id})is not in the source cell ({source_cell.cell_id})"
        assert (
            ue.ue_imsi in source_cell.connected_ue_list
        ), "UE is not connected to the source cell"
        assert (
            ue.ue_imsi not in target_cell.connected_ue_list
        ), "UE is already connected to the target cell"

        source_bs = source_cell.base_station
        target_bs = target_cell.base_station

        if source_bs.bs_id == target_bs.bs_id:
            # same base station, just change the cell
            target_cell.register_ue(ue)
            ue.execute_handover(target_cell)
            self.ue_registry[ue.ue_imsi]["cell"] = target_cell
            source_cell.deregister_ue(ue)
            logger.info(
                f"gNB {self.bs_id}: Handover UE {ue.ue_imsi} from cell {source_cell.cell_id} to cell {target_cell.cell_id}"
            )
        else:
            ue_reg_data = source_bs.ue_registry[ue.ue_imsi].copy()
            ue_reg_data["cell"] = target_cell
            ue_reg_data["rrc_meas_events"] = target_bs.rrc_measurement_events.copy()
            target_bs.ue_registry[ue.ue_imsi] = ue_reg_data
            target_cell.register_ue(ue)
            ue.execute_handover(target_cell)
            source_cell.deregister_ue(ue)
            del source_bs.ue_registry[ue.ue_imsi]
            logger.info(
                f"gNB {self.bs_id} Handover UE {ue.ue_imsi} from cell {source_cell.cell_id} to BS: {target_bs.bs_id} cell {target_cell.cell_id} (different BS)"
            )

    def step(self, delta_time):
        # first update cell first
        for cell in self.cell_list.values():
            cell.step(delta_time)

        # reset RIC control actions
        self.ric_control_actions = []

        # process RRC measurement events
        while len(self.ue_rrc_meas_events) > 0:
            event = self.ue_rrc_meas_events.pop(0)
            event_id = event["event_id"]
            if event_id not in self.ue_rrc_meas_event_handers:
                logger.info(
                    f"gNB {self.bs_id}: No handler for event ID {event_id}. Skipping."
                )
                continue
            handler = self.ue_rrc_meas_event_handers[event_id]
            action = handler(event)

            if action is not None:
                # add the action to the RIC control actions list
                self.ric_control_actions.append(action)

            logger.info(
                f"gNB {self.bs_id}: Processed RRC measurement event {event_id} for UE {event["triggering_ue"].ue_imsi}"
            )

        # process (reject, merge or execute) all the RIC control actions
        self.process_ric_control_actions()

    def on_ue_application_traffic(self, ue, traffic_data):
        url = traffic_data["url"]

        # for the moment, we only support the AI service traffic
        if not url.startswith("http://cranfield_6G.com/ai_services/"):
            return

        # when a connected UE requests edge AI service
        ai_service_name = url.replace("http://cranfield_6G.com/ai_services/", "")
        ue_imsi = traffic_data["data"]["ue_id"]
        if not ai_service_name:
            logger.warning("Undefined ai_service_name")
            return

        # local breakout
        ai_service_subscription = self.edge_server.check_ue_subscription(
            ai_service_name, ue_imsi
        )
        if not ai_service_subscription:
            return

        # forward the request to the edge server
        start_time = time.time() * 1000  # convert to milliseconds
        response = self.edge_server.handle_ai_service_request(
            ai_service_subscription=ai_service_subscription,
            request_data=traffic_data["data"],
            request_files=traffic_data.get("files", {}),
        )

        end_time = time.time() * 1000  # convert to milliseconds

        if self.ai_service_event_handler:
            files = traffic_data.get("files", {})
            request_files_size = 0
            if files and files.get("file", None):
                request_files_size = len(files["file"])
                # encode the files from bytes to base64 string
                files["file_base64"] = base64.b64encode(files["file"]).decode("utf-8")
                del files["file"]
            self.ai_service_event_handler(
                {
                    "ue_imsi": ue.ue_imsi,
                    "request": {
                        "ai_service_name": ai_service_name,
                        "ue_imsi": ue_imsi,
                        "request_data": traffic_data["data"],
                        "request_files": files,
                        "request_files_size": request_files_size,
                    },
                    "response": response,
                    "service_response_time_ms": end_time - start_time,
                }
            )

        return response
