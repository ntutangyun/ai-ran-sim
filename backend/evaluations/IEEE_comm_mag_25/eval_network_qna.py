# the evaluation is done manually through the web interface for the moment.
# the ground truth knowledge retrieval path is also set manually

# evaluation criteria:

# answer_accuracy:
#   1.0 if the final answer is correct and complete.
#   0.5 if the final answer is correct but incomplete (e.g., missing some related knowledge).
#   0.0 if any part of the final answer is incorrect or no answer is provided.
# tool_efficiency:
#   1.0 if the agent follows the ground truth chains of tool calls.
#   0.5 if the agent there are abundant tool calls.
#   0.0 if the agent does not use any tool.
# tool_accuracy:
#   1.0 if the agent uses the correct tool to retrieve the knowledge.
#   0.5 if for some knowledge queries, the knowledge endpoint is not supported but the attributes exists in the code.
#   0.0 if some of the knowledge endpoints are incorrect.

# instructions:
#   There can be multiple ground truth chains of knowledge tool calls for each question.
#   reset after each conversation.
#   pause the simulation after a few seconds before asking the questions particularly for live network data Q&A.
# evaluation done at 10/07/2025 using GPT-4.1 API.
simple_qna_conversations = [
    {
        "answer_accuracy": 1.0,
        "tool_efficiency": 1.0,
        "tool_accuracy": 1.0,
        "ground_truth_conversations": [
            [
                {
                    "type": "user",
                    "content": "how many user equipments are connected to the network?",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/docs/user_equipments",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/user_equipments",
                },
                {
                    "type": "final_answer",
                    "content": 18,
                },
            ]
        ],
        "actual_conversation": [
            {
                "type": "user",
                "content": "how many user equipments are connected to the network?",
            },
            {
                "type": "agent_activation",
                "agent_name": "Basic Network Knowledge Assistant",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/user_equipments",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/user_equipments",
            },
            {
                "type": "answer",
                "content": "There are currently 18 user equipments (UEs) connected to the network.",
            },
        ],
    },
    {
        "answer_accuracy": 1.0,
        "tool_efficiency": 1.0,
        "tool_accuracy": 1.0,
        "ground_truth_conversations": [
            [
                {
                    "type": "user",
                    "content": "how many cells are connected to the network?",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/docs/cells",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/cells",
                },
                {
                    "type": "final_answer",
                    "content": 8,
                },
            ]
        ],
        "actual_conversation": [
            {"type": "user", "content": "how many cells are connected to the network?"},
            {
                "type": "agent_activation",
                "agent_name": "Basic Network Knowledge Assistant",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/cells",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/cells",
            },
            {
                "type": "answer",
                "content": "There are 8 cells in the simulated network.",
            },
        ],
    },
    {
        "answer_accuracy": 1.0,
        "tool_efficiency": 1.0,
        "tool_accuracy": 1.0,
        "ground_truth_conversations": [
            [
                {
                    "type": "user",
                    "content": "how many base stations in the simulated network?",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/docs/base_stations",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/base_stations",
                },
                {
                    "type": "final_answer",
                    "content": 4,
                },
            ]
        ],
        "actual_conversation": [
            {
                "type": "user",
                "content": "how many base stations in the simulated network?",
            },
            {
                "type": "agent_activation",
                "agent_name": "Basic Network Knowledge Assistant",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/base_stations",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/base_stations",
            },
            {
                "type": "answer",
                "content": "There are 4 base stations in the simulated network. Their identifiers are: bs_11, bs_12, bs_21, and bs_22.",
            },
        ],
    },
    {
        "answer_accuracy": 1.0,
        "tool_efficiency": 1.0,
        "tool_accuracy": 1.0,
        "ground_truth_conversations": [
            [
                {
                    "type": "user",
                    "content": "which cell is the User equipment of ID: IMSI_1 connected to?",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/docs/user_equipments",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/user_equipments/IMSI_1/attributes/current_cell",
                },
                {
                    "type": "final_answer",
                    "content": "bs_21_cell_mid_freq",
                },
            ],
            [
                {
                    "type": "user",
                    "content": "which cell is the User equipment of ID: IMSI_1 connected to?",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/docs/user_equipments",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/user_equipments/IMSI_1",
                },
                {
                    "type": "final_answer",
                    "content": "bs_21_cell_mid_freq",
                },
            ],
        ],
        "actual_conversations": [
            {
                "type": "user",
                "content": "which cell is the User equipment of ID: IMSI_1 connected to?",
            },
            {
                "type": "agent_activation",
                "agent_name": "Basic Network Knowledge Assistant",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/user_equipments",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/user_equipments/IMSI_1/attributes/current_cell",
            },
            {
                "type": "answer",
                "content": 'The User Equipment with ID: IMSI_1 is currently connected to the cell "bs_21_cell_mid_freq," which belongs to base station "bs_21" and operates on frequency band n78 with a carrier frequency of 3500 MHz.',
            },
        ],
    },
    {
        "answer_accuracy": 1.0,
        "tool_efficiency": 1.0,
        "tool_accuracy": 1.0,
        "ground_truth_conversations": [
            [
                {
                    "type": "user",
                    "content": "what's the current load of cell of ID: bs_11_cell_mid_freq?",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/docs/cells",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/cells/bs_11_cell_mid_freq",
                },
                {
                    "type": "final_answer",
                    "content": 0.1245,
                },
            ],
            [
                {
                    "type": "user",
                    "content": "what's the current load of cell of ID: bs_11_cell_mid_freq?",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/cells/bs_11_cell_mid_freq",
                },
                {
                    "type": "final_answer",
                    "content": 0.1245,
                },
            ],
            [
                {
                    "type": "user",
                    "content": "what's the current load of cell of ID: bs_11_cell_mid_freq?",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/docs/cells",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/cells/bs_11_cell_mid_freq/attributes/current_load",
                },
                {
                    "type": "final_answer",
                    "content": 0.1245,
                },
            ],
        ],
        "actual_conversations": [
            {
                "type": "user",
                "content": "what's the current load of cell of ID: bs_11_cell_mid_freq?",
            },
            {
                "type": "agent_activation",
                "agent_name": "Basic Network Knowledge Assistant",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/cells/bs_11_cell_mid_freq",
            },
            {
                "type": "final_answer",
                "content": """The current load of the cell with ID bs_11_cell_mid_freq is approximately 0.12 (or 12.45%). This value represents the proportion of the cell's total capacity currently being utilized. 

- Downlink (DL) load: 15.6%
- Uplink (UL) load: 0%
- 2 user equipments (UEs) are connected to this cell.

Let me know if you need a breakdown by user or more technical details!""",
            },
        ],
    },
]


complex_qna_conversations = [
    {
        "answer_accuracy": 1.0,
        "tool_efficiency": 0.5,
        "tool_accuracy": 1.0,
        "ground_truth_conversations": [
            [
                {
                    "type": "user",
                    "content": "explain what happens when an UE powers up with code-level reference.",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/docs/user_equipments",
                },
                {
                    "type": "tool_get_knowledge_bulk",
                    "query_key": [
                        "/docs/user_equipments/methods/power_up",
                        "/docs/user_equipments/methods/monitor_signal_strength",
                        "/docs/user_equipments/methods/cell_selection_and_camping",
                        "/docs/user_equipments/methods/authenticate_and_register",
                    ],
                },
                {
                    "type": "final_answer",
                    "content": "explain with code the essential steps: 1) monitor_signal_strength, 2) cell_selection_and_camping, 3) authenticate_and_register.",
                },
            ]
        ],
        "actual_conversation": [
            {
                "type": "user",
                "content": "explain what happens when an UE powers up with code-level reference.",
            },
            {
                "type": "agent_activation",
                "agent_name": "Basic Network Knowledge Assistant",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/user_equipments",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/base_stations",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/cells",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/ric",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/sim_engine",
            },
            {
                "type": "tool_get_knowledge_bulk",
                "query_key": [
                    "/docs/user_equipments/methods/power_up",
                    "/docs/user_equipments/methods/monitor_signal_strength",
                    "/docs/user_equipments/methods/cell_selection_and_camping",
                    "/docs/user_equipments/methods/authenticate_and_register",
                ],
            },
            {
                "type": "final_answer",
                "content": """
When a User Equipment (UE) powers up in the network simulation, the following steps are executed at the code level:

---

### 1. **UE Power-Up**  
```python
def power_up(self):
    print(f"UE {self.ue_imsi} Powering up")
    self.monitor_signal_strength()

    if len(list(self.downlink_received_power_dBm_dict.values())) == 0:
        print(f"UE {self.ue_imsi}: No cells detected. Powering down...")
        return False

    if not self.cell_selection_and_camping():
        print(f"UE {self.ue_imsi}: Cell selection and camping failed.")
        return False

    if not self.authenticate_and_register():
        print(f"UE {self.ue_imsi}: Authentication and registration failed.")
        return False

    self.connected = True
    return True
```
- **What Happens:**  
  The UE begins by monitoring for available signals, then attempts to select and connect to a suitable cell, and finally authenticates and registers with the network. If any step fails, the process stops.

---

### 2. **Signal Strength Monitoring**  
```python
def monitor_signal_strength(self):
    ...
    # Scan all cells by calculating downlink received power
    for cell in self.simulation_engine.cell_list.values():
        ...
        # Estimate received power, check thresholds, build list of detected cells
    ...
    self.calculate_SINR_and_CQI()
    return True
```
- **What Happens:**  
  The UE scans available cells, measures the power of their signals, and builds a list of detectable cells. It also calculates quality metrics (SINR and CQI).

---

### 3. **Cell Selection and Camping**  
```python
def cell_selection_and_camping(self):
    # Rank detected cells by frequency priority, then received power (with offset)
    cells_detected.sort(key=lambda x: (x["frequency_priority"], x["received_power_with_cio_dBm"]), reverse=True)
    ...
    self.set_current_cell(cells_detected[0]["cell"])
    return True
```
- **What Happens:**  
  Detected cells are ranked. The UE chooses the cell with the best signal and highest priority, then “camps” on it—making it the current serving cell.

---

### 4. **Authentication and Registration**  
```python
def authenticate_and_register(self):
    if self.current_bs is None:
        print("No base station to authenticate and register with.")
        return False
    ue_reg_res = self.current_bs.handle_ue_authentication_and_registration(self)
    self.slice_type = ue_reg_res["slice_type"]
    self.qos_profile = ue_reg_res["qos_profile"]
    self.setup_rrc_measurement_event_monitors(ue_reg_res["rrc_meas_events"])
    return True
```
- **What Happens:**  
  The UE authenticates and registers with its selected serving cell’s base station. Here, it is assigned a network slice and a QoS profile. RRC measurement event monitors are also set up for mobility and link quality management.

---

#### **Summary Table**

| Step                          | Method                           | Purpose                                    |
|-------------------------------|----------------------------------|--------------------------------------------|
| Power Up                      | `power_up()`                     | Orchestrates initial connection sequence   |
| Scan Signal Strength          | `monitor_signal_strength()`      | Finds cells and assesses signal quality    |
| Select & Camp on Cell         | `cell_selection_and_camping()`   | Picks best cell and switches to it         |
| Authenticate & Register       | `authenticate_and_register()`    | Integrates with network and sets QoS/slice |

---

**Related code methods**: 
- `/docs/user_equipments/methods/power_up`
- `/docs/user_equipments/methods/monitor_signal_strength`
- `/docs/user_equipments/methods/cell_selection_and_camping`
- `/docs/user_equipments/methods/authenticate_and_register`

Would you like expanded code examples or details about error handling in these steps?

""",
            },
        ],
    },
    {
        "answer_accuracy": 0.5,  # missing the xApp part for the single test.
        "tool_efficiency": 1,  # used the bulk query tool in multiple rounds of knowledge query.
        "tool_accuracy": 0.5,  # queried attributes that are unsupported at the moment, although referenced by other supported endpoints.
        "ground_truth_conversations": [
            [
                {
                    "type": "user",
                    "content": "explain the complete handover mechanism implementation in the simulated network with code-level reference.",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge_bulk",
                    "query_key": [
                        "/docs/user_equipments",
                        "/docs/cells",
                        "/docs/base_stations",
                        "/docs/ric",
                        "/docs/sim_engine",
                    ],
                },
                {
                    "type": "tool_get_knowledge_bulk",
                    "query_key": [
                        "/docs/user_equipments/methods/setup_rrc_measurement_event_monitors",
                        "/docs/user_equipments/methods/power_up",
                        "/docs/user_equipments/methods/monitor_signal_strength",
                        "/docs/user_equipments/methods/cell_selection_and_camping",
                        "/docs/user_equipments/methods/authenticate_and_register",
                        "/docs/user_equipments/methods/execute_handover",
                        "/docs/user_equipments/methods/set_current_cell",
                        "/docs/user_equipments/methods/check_rrc_meas_events_to_monitor",
                        "/docs/user_equipments/methods/step",
                        "/docs/base_stations/methods/receive_ue_rrc_meas_events",
                        "/docs/base_stations/methods/init_rrc_measurement_event_handler",
                        "/docs/base_stations/methods/process_ric_control_actions",
                        "/docs/base_stations/methods/execute_handover",
                        "/docs/base_stations/methods/step",
                        "/docs/cells/methods/deregister_ue",
                        "/docs/cells/methods/register_ue",
                        "/docs/ric/xapps",
                        "/docs/ric/xapps/xAppA3HandoverBlind",
                    ],
                },
                {
                    "type": "final_answer",
                    "content": "explain with code the essential steps: 1) RRC event monitoring and triggering by UE, 2) base station queues and handles the RRC events, 3) xApp in RIC processes the RRC event and generates handover command/action/control to base station 4) base station performs the handover action. and 5) source and target cell updates the UE list.",
                },
            ]
        ],
        "actual_conversation": [
            {
                "type": "user",
                "content": "explain the complete handover mechanism implementation in the simulated network with code-level reference.",
            },
            {
                "type": "agent_activation",
                "agent_name": "Basic Network Knowledge Assistant",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/user_equipments",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/base_stations",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/cells",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/ric",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/sim_engine",
            },
            {
                "type": "tool_get_knowledge_bulk",
                "query_key": [
                    "/docs/user_equipments/methods/execute_handover",
                    "/docs/user_equipments/methods/setup_rrc_measurement_event_monitors",
                    "/docs/user_equipments/methods/check_rrc_meas_events_to_monitor",
                    "/docs/user_equipments/attributes/current_cell",
                ],
            },
            {
                "type": "tool_get_knowledge_bulk",
                "query_key": [
                    "/docs/base_stations/methods/execute_handover",
                    "/docs/base_stations/methods/receive_ue_rrc_meas_events",
                    "/docs/base_stations/attributes/ue_rrc_meas_events",
                    "/docs/base_stations/attributes/ue_rrc_meas_event_handlers",
                ],
            },
            {
                "type": "tool_get_knowledge_bulk",
                "query_key": [
                    "/docs/cells/methods/register_ue",
                    "/docs/cells/methods/deregister_ue",
                    "/docs/cells/attributes/connected_ue_list",
                ],
            },
            {
                "type": "tool_get_knowledge_bulk",
                "query_key": ["/docs/sim_engine/methods/step"],
            },
            {
                "type": "tool_get_knowledge_bulk",
                "query_key": [
                    "/docs/user_equipments/methods/set_current_cell",
                    "/docs/user_equipments/attributes/rrc_measurement_event_monitors",
                    "/docs/user_equipments/attributes/serving_cell_history",
                    "/docs/user_equipments/attributes/downlink_received_power_dBm_dict",
                    "/docs/user_equipments/attributes/downlink_sinr",
                    "/docs/user_equipments/attributes/downlink_cqi",
                    "/docs/user_equipments/attributes/downlink_mcs_index",
                    "/docs/user_equipments/attributes/downlink_mcs_data",
                    "/docs/user_equipments/attributes/downlink_bitrate",
                    "/docs/user_equipments/attributes/downlink_latency",
                    "/docs/user_equipments/attributes/uplink_bitrate",
                    "/docs/user_equipments/attributes/uplink_latency",
                    "/docs/user_equipments/attributes/uplink_transmit_power_dBm",
                ],
            },
            {
                "type": "tool_get_knowledge_bulk",
                "query_key": ["/docs/base_stations/attributes/ue_registry"],
            },
            {
                "type": "tool_get_knowledge_bulk",
                "query_key": [
                    "/docs/base_stations/methods/process_ric_control_actions"
                ],
            },
            {
                "type": "final_answer",
                "content": """
Here is a detailed, code-level explanation of how the handover mechanism is implemented in the simulated telecom network, based on the network knowledge base:

---

### **1. Handover Overview in the Simulation**

**Handover** is the process that enables a User Equipment (UE) to switch its connection from a source cell to a target cell, usually due to changes in radio conditions, mobility, or load balancing.

The handover mechanism involves the following entities and their interactions:
- **UE (User Equipment)**: Monitors radio conditions and reports measurement events.
- **Cell**: Offers resources to UEs and maintains a list of connected users.
- **Base Station (BS)**: Manages handover procedures and registration of UEs.
- **RIC (RAN Intelligent Controller)**: Can trigger/control handover actions.
- **Simulation Engine**: Orchestrates the time-stepped simulation in which all operations (including handovers) happen.

---

### **2. Key Components and Attributes Involved**

**User Equipment (UE):**
- Attributes: `current_cell`, `downlink_received_power_dBm_dict`, `downlink_sinr`, `downlink_cqi`, `downlink_mcs_index`, `serving_cell_history`, etc.
- Methods: `execute_handover(target_cell)`, `setup_rrc_measurement_event_monitors(...)`, `check_rrc_meas_events_to_monitor()`, `set_current_cell(cell)`

**Base Station (BS):**
- Attributes: `ue_registry`, `ue_rrc_meas_events`, `ric_control_actions`
- Methods: `execute_handover(ue, source_cell, target_cell)`, `receive_ue_rrc_meas_events(event)`, `process_ric_control_actions()`

**Cell:**
- Attributes: `connected_ue_list`
- Methods: `register_ue(ue)`, `deregister_ue(ue)`

---

### **3. Handover Event Flow / Code-level Mechanism**

#### **A. Monitoring and Triggering Handover via RRC Measurement Events**
UEs set up RRC event monitors (e.g., for A3 event: neighbor signal better than serving cell by a threshold). This setup is performed via `setup_rrc_measurement_event_monitors`.

**Snippet:**
```python
def setup_rrc_measurement_event_monitors(self, rrc_meas_events_to_monitor=[]):
    self.rrc_measurement_event_monitors = [
        get_rrc_measurement_event_monitor(event["event_id"], event_params=event)
        for event in rrc_meas_events_to_monitor
    ]
```

Each step (simulation tick), the UE checks if any configured event is triggered:
```python
def check_rrc_meas_events_to_monitor(self):
    ...
    for rrc_meas_event_trigger in self.rrc_measurement_event_monitors:
        rrc_meas_event_trigger.check(self, cell_signal_map.copy())
        if rrc_meas_event_trigger.is_triggered:
            event_report = rrc_meas_event_trigger.gen_event_report()
            self.current_bs.receive_ue_rrc_meas_events(event_report)
```
- If triggered, an event report is sent to the UE's serving BS.

#### **B. Base Station Receives and Queues Handover Events**
On receiving a measurement event, BS queues it for processing:
```python
def receive_ue_rrc_meas_events(self, event):
    ...
    self.ue_rrc_meas_events.append(event)
```
This list (`ue_rrc_meas_events`) consists of events reported in the current or past steps.

#### **C. BS and RIC Orchestrate Handover Actions**
BS, often under direction from the RIC, processes pending control actions (predominantly handovers) each step:
```python
def process_ric_control_actions(self):
    ...
    for ue_imsi, actions in ue_handover_actions.items():
        action = actions[0]
        ue = action.action_data["ue"]
        source_cell = self.simulation_engine.cell_list[action.action_data["source_cell_id"]]
        target_cell = self.simulation_engine.cell_list[action.action_data["target_cell_id"]]
        self.execute_handover(ue, source_cell, target_cell)
        break
```
- The BS fetches source/target cell objects and invokes `execute_handover`.

#### **D. Performing the Handover**

**On the BS:**
```python
def execute_handover(self, ue, source_cell, target_cell):
    ...
    if source_bs.bs_id == target_bs.bs_id:
        # same base station, just change cell
        target_cell.register_ue(ue)
        ue.execute_handover(target_cell)
        self.ue_registry[ue.ue_imsi]["cell"] = target_cell
        source_cell.deregister_ue(ue)
    else:
        # inter-BS handover
        ue_reg_data = source_bs.ue_registry[ue.ue_imsi].copy()
        ue_reg_data["cell"] = target_cell
        target_bs.ue_registry[ue.ue_imsi] = ue_reg_data
        target_cell.register_ue(ue)
        ue.execute_handover(target_cell)
        source_cell.deregister_ue(ue)
        del source_bs.ue_registry[ue.ue_imsi]
```
- UE is added to target cell (`register_ue`), removed from source cell (`deregister_ue`)
- UE's registry entry is updated from source BS to target BS (for inter-BS handover)
- Triggers UE's own `execute_handover`

**On the UE:**
```python
def execute_handover(self, target_cell):
    self.downlink_received_power_dBm_dict = {}
    self.set_downlink_sinr(0)
    ...
    self.set_current_cell(target_cell)
    for event_monitor in self.rrc_measurement_event_monitors:
        event_monitor.reset_trigger_history()
```
- All radio stats are reset
- The UE's `current_cell` is updated via `set_current_cell`, and handover history (`serving_cell_history`) appended
- Measurement event monitors reset trigger history for the new cell context

#### **E. Cell Registration and Resource Management**
**Cell registration/deregistration:**
```python
def register_ue(self, ue):
    self.connected_ue_list[ue.ue_imsi] = ue
    self.prb_ue_allocation_dict[ue.ue_imsi] = {"downlink": 0, "uplink": 0}

def deregister_ue(self, ue):
    if ue.ue_imsi in self.prb_ue_allocation_dict:
        del self.prb_ue_allocation_dict[ue.ue_imsi]
    if ue.ue_imsi in self.connected_ue_list:
        del self.connected_ue_list[ue.ue_imsi]
```
- Target cell's resource pool updated to add UE
- Source cell's resource pool updated to release resources for departing UE

---

### **4. Orchestration via Simulation Engine**

The **simulation engine** (`step` method) coordinates the above by advancing the simulation, including calling each UE and BS's step logic (which would include signal monitoring, RRC event checking, handover actions, etc.).

---

### **5. Summary Diagram**

**Handover flow:**
```
[UE monitors signal] --> [UE triggers RRC event] --> [BS receives event] --> 
[RIC/BS decides handover] --> [BS executes handover: re-registers UE, updates cells] --> 
[UE resets and associates to new cell]
```

---

### **References to Implementation**

- Measurement and RRC trigger: `setup_rrc_measurement_event_monitors`, `check_rrc_meas_events_to_monitor`
- UE handover execution: `execute_handover(self, target_cell)`, `set_current_cell`
- BS handover handling: `execute_handover(ue, source_cell, target_cell)`
- Cell registration: `register_ue`, `deregister_ue`
- Event orchestration: Simulation engine's `step`.

If you need **actual code snippets** for a particular method or a detailed walkthrough for a specific step, let me know!
""",
            },
        ],
    },
]

# In intuitive reuse conversations, the agent will be involved in two rounds of conversation,
# and the agent is expected to reuse the knowledge query results from the first round in the second round to infer the knowledge query key.
intuitive_reuse_conversations = [
    {
        "reuse_validated": True,
        "ground_truth_conversations": [
            [
                {
                    "type": "user",
                    "content": "what's the current downlink bitrate of User Equipment with ID: IMSI_1?",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/docs/user_equipments",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/user_equipments/IMSI_1/attributes/downlink_bitrate",
                },
                {
                    "type": "final_answer",
                    "content": 10110294.11764706,
                },
                {
                    "type": "user",
                    "content": "what's the `position_x` of base station of ID: bs_12? Try to infer the knowledge query directly.",
                },
                {
                    "type": "agent_activation",
                    "agent_name": "Basic Network Knowledge Assistant",
                },
                {
                    "type": "tool_get_knowledge",
                    "query_key": "/base_stations/bs_12/attributes/position_x",
                },
                {
                    "type": "final_answer",
                    "content": 1500,
                },
            ],
        ],
        "actual_conversation": [
            {
                "type": "user",
                "content": "what's the current downlink bitrate of User Equipment with ID: IMSI_1?",
            },
            {
                "type": "agent_activation",
                "agent_name": "Basic Network Knowledge Assistant",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/docs/user_equipments",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/user_equipments/IMSI_1/attributes/downlink_bitrate",
            },
            {
                "type": "final_answer",
                "content": "The current downlink bitrate of User Equipment with ID: IMSI_1 is approximately 10,294,118 bps (bits per second).",
            },
            {
                "type": "user",
                "content": "what's the `position_x` of base station of ID: bs_12? Try to infer the knowledge query directly.",
            },
            {
                "type": "agent_activation",
                "agent_name": "Basic Network Knowledge Assistant",
            },
            {
                "type": "tool_get_knowledge",
                "query_key": "/base_stations/bs_12/attributes/position_x",
            },
            {
                "type": "final_answer",
                "content": "The position_x of the base station with ID: bs_12 is 1500.",
            },
        ],
    }
]
