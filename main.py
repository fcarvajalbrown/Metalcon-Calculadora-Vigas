import sys
import numpy as np  # For numerical operations and matrix math
from datetime import datetime  # For timestamping exported reports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QSpinBox, 
                             QPushButton, QTextEdit, QGroupBox, QDoubleSpinBox,
                             QSplitter, QFormLayout, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter  # For print functionality
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.pyplot as plt  # For creating plots and diagrams

# METALCON C profiles from the Cintac catalog (page 4)
# Each profile has: A = height (mm), B = width (mm), e = thickness (mm), 
# weight = kg/m, I = moment of inertia (mm^4)
PROFILES = {
    '40CA085': {'A': 40, 'B': 38, 'e': 0.85, 'weight': 0.83, 'I': 2.1e4},
    '60CA085': {'A': 60, 'B': 38, 'e': 0.85, 'weight': 0.96, 'I': 4.8e4},
    '90CA085': {'A': 90, 'B': 38, 'e': 0.85, 'weight': 1.23, 'I': 1.1e5},
    '100CA085': {'A': 100, 'B': 40, 'e': 0.85, 'weight': 1.32, 'I': 1.4e5},
    '150CA085': {'A': 150, 'B': 40, 'e': 0.85, 'weight': 1.64, 'I': 3.2e5},
    '90CA10': {'A': 90, 'B': 38, 'e': 1.0, 'weight': 1.44, 'I': 1.3e5},
    '150CA10': {'A': 150, 'B': 40, 'e': 1.0, 'weight': 1.94, 'I': 3.8e5},
    '150CA16': {'A': 150, 'B': 40, 'e': 1.6, 'weight': 3.06, 'I': 5.9e5},
}

class BeamSolver:
    """
    Finite Element Analysis solver for simple beams with multiple supports.
    
    Uses Euler-Bernoulli beam theory to calculate:
    - Deflections (vertical displacement)
    - Rotations (slope of the beam)
    - Shear forces
    - Bending moments
    - Support reactions
    """
    
    def __init__(self, length, supports, loads, profile, E=200000):
        """
        Initialize the beam solver with geometric and loading parameters.
        
        Args:
            length (float): Total beam length in meters
            supports (list): List of (position, type) tuples where:
                - position is in meters from left end
                - type is either 'pinned' (fixed in x,y) or 'roller' (fixed in y only)
            loads (list): List of (position, force) tuples where:
                - position is in meters from left end
                - force is in Newtons (negative = downward)
            profile (dict): METALCON profile dictionary with geometric properties
            E (float): Young's modulus in MPa (default 200000 for steel)
        """
        self.L = length  # Store beam length
        self.supports = sorted(supports, key=lambda x: x[0])  # Sort supports left to right
        self.loads = loads  # Store load list
        self.profile = profile  # Store profile properties
        self.E = E * 1e6  # Convert Young's modulus from MPa to Pa for calculations
        
        # Calculate moment of inertia and convert from mm^4 to m^4
        # Moment of inertia (I) measures resistance to bending
        self.I = profile['I'] / 1e12
        
        # Divide beam into 50 elements for finite element analysis
        # More elements = more accurate but slower computation
        self.n_elements = 50
        
    def solve(self):
        """
        Solve the beam using the Finite Element Method (FEM).
        
        Process:
        1. Discretize beam into elements
        2. Assemble global stiffness matrix [K]
        3. Assemble load vector {F}
        4. Apply boundary conditions (supports)
        5. Solve [K]{u} = {F} for displacements {u}
        6. Calculate derived quantities (shear, moment, reactions)
        
        Returns:
            dict: Contains x positions, deflection, rotation, shear, moment, and reactions
        """
        n_elem = self.n_elements  # Number of beam elements
        n_nodes = n_elem + 1  # Number of nodes (always 1 more than elements)
        elem_length = self.L / n_elem  # Length of each element in meters
        
        # Create array of node positions along the beam
        x_nodes = np.linspace(0, self.L, n_nodes)
        
        # Each node has 2 degrees of freedom (DOF):
        # - Vertical displacement (w)
        # - Rotation (theta)
        n_dof = n_nodes * 2
        
        # Initialize global stiffness matrix [K] and force vector {F}
        # [K] relates displacements to forces: [K]{u} = {F}
        K = np.zeros((n_dof, n_dof))
        F = np.zeros(n_dof)
        
        # STEP 1: Assemble global stiffness matrix using Euler-Bernoulli beam elements
        # For each element, calculate local stiffness and add to global matrix
        for i in range(n_elem):
            Le = elem_length  # Element length
            EI = self.E * self.I  # Flexural rigidity (resistance to bending)
            
            # Element stiffness matrix for Euler-Bernoulli beam
            # This is a 4x4 matrix relating the 4 DOFs of a 2-node beam element
            # DOFs: [w1, theta1, w2, theta2] where w=displacement, theta=rotation
            ke = (EI / Le**3) * np.array([
                [12,      6*Le,    -12,     6*Le],      # Row 1: Forces/moments at node 1
                [6*Le,    4*Le**2, -6*Le,   2*Le**2],   # Row 2: due to displacements
                [-12,     -6*Le,   12,      -6*Le],     # Row 3: at nodes 1 and 2
                [6*Le,    2*Le**2, -6*Le,   4*Le**2]    # Row 4:
            ])
            
            # Map local element DOFs to global DOFs
            # Node i has DOFs [2*i, 2*i+1], Node i+1 has DOFs [2*(i+1), 2*(i+1)+1]
            dofs = [i*2, i*2+1, (i+1)*2, (i+1)*2+1]
            
            # Add element stiffness to global stiffness matrix
            # This is called "assembly" in FEM
            for ii in range(4):
                for jj in range(4):
                    K[dofs[ii], dofs[jj]] += ke[ii, jj]
        
        # STEP 2: Apply point loads to force vector
        # Distribute each load to the nearest node
        for pos, force in self.loads:
            # Find the closest node to the load position
            node_idx = int(round(pos / self.L * n_elem))
            # Ensure node index is within valid range
            if 0 <= node_idx < n_nodes:
                # Add force to the vertical displacement DOF (even indices)
                F[node_idx * 2] += force
        
        # STEP 3: Apply boundary conditions (supports)
        # Boundary conditions constrain certain DOFs to zero displacement
        fixed_dofs = []  # List to store which DOFs are fixed
        
        for pos, support_type in self.supports:
            # Find the node closest to the support position
            node_idx = int(round(pos / self.L * n_elem))
            if 0 <= node_idx < n_nodes:
                # All supports fix vertical displacement (w)
                fixed_dofs.append(node_idx * 2)
                
                # Pinned supports also prevent rotation (theta)
                # Roller supports allow rotation
                if support_type == 'pinned':
                    fixed_dofs.append(node_idx * 2 + 1)
        
        # Modify the system to enforce boundary conditions
        # We use the "penalty method": make diagonal entries very large
        K_mod = K.copy()  # Create a copy to avoid modifying original
        F_mod = F.copy()
        
        for dof in fixed_dofs:
            # Zero out the row and column for this DOF
            K_mod[dof, :] = 0
            K_mod[:, dof] = 0
            # Set diagonal to 1 and force to 0 to enforce zero displacement
            K_mod[dof, dof] = 1
            F_mod[dof] = 0
        
        # STEP 4: Solve the linear system [K]{u} = {F} for displacements
        try:
            U = np.linalg.solve(K_mod, F_mod)  # Direct solver for linear system
        except np.linalg.LinAlgError:
            # If matrix is singular (shouldn't happen with proper BCs), return zeros
            U = np.zeros(n_dof)
        
        # STEP 5: Extract results from displacement vector
        # Separate vertical displacements (w) and rotations (theta)
        w = U[::2]  # Every even index: vertical displacements
        theta = U[1::2]  # Every odd index: rotations
        
        # STEP 6: Calculate support reactions
        # Reaction = (K * u - F) at support DOFs
        reactions = []
        for pos, support_type in self.supports:
            node_idx = int(round(pos / self.L * n_elem))
            if 0 <= node_idx < n_nodes:
                # Calculate reaction force at this support
                # Multiply stiffness matrix row by displacement vector
                reaction = 0
                for j in range(n_dof):
                    reaction += K[node_idx * 2, j] * U[j]
                # Subtract applied force at this node
                reaction -= F[node_idx * 2]
                # Store with negative sign (reaction acts upward if load is downward)
                reactions.append((pos, -reaction))
        
        # STEP 7: Calculate shear force and bending moment diagrams
        # These are derived from equilibrium equations
        shear = np.zeros(n_nodes)  # Shear force at each node
        moment = np.zeros(n_nodes)  # Bending moment at each node
        
        for i in range(n_nodes):
            x = x_nodes[i]  # Current position along beam
            
            # Shear force: Sum of all forces to the left of this point
            # Start with reactions (positive upward)
            for pos, reaction in reactions:
                if pos < x:  # Only count reactions to the left
                    shear[i] += reaction
            
            # Subtract applied loads to the left (negative forces are downward)
            for pos, force in self.loads:
                if pos < x:
                    shear[i] += force  # Add (force is already negative for down)
            
            # Bending moment: Sum of moments about this point from all forces to the left
            # Moment = Force × Distance
            for pos, reaction in reactions:
                if pos < x:
                    moment[i] += reaction * (x - pos)
            
            for pos, force in self.loads:
                if pos < x:
                    moment[i] += force * (x - pos)
        
        # Return all results as a dictionary
        return {
            'x': x_nodes,  # Position along beam (m)
            'deflection': w * 1000,  # Convert m to mm for readability
            'rotation': theta,  # Rotation in radians
            'shear': shear / 1000,  # Convert N to kN for readability
            'moment': moment / 1000,  # Convert N⋅m to kN⋅m for readability
            'reactions': reactions  # List of (position, force) tuples
        }

class MplCanvas(FigureCanvas):
    """
    Custom Matplotlib canvas widget for embedding plots in PyQt5.
    
    This creates a Qt widget that can display matplotlib figures,
    allowing seamless integration of plots into the GUI.
    """
    
    def __init__(self, parent=None):
        """Initialize the canvas with a blank figure."""
        # Create a new matplotlib figure with white background
        self.fig = Figure(figsize=(10, 8), facecolor='white')
        # Initialize the FigureCanvas base class
        super().__init__(self.fig)
        # Set the Qt parent widget
        self.setParent(parent)

class MainWindow(QMainWindow):
    """
    Main application window containing all UI elements and logic.
    
    This class manages:
    - User input controls (geometry, loads, supports, profile)
    - Visualization canvas
    - Analysis execution
    - PDF export and printing
    """
    
    def __init__(self):
        """Initialize the main window and set up the UI."""
        super().__init__()
        self.setWindowTitle("METALCON Simple Beam Solver")
        self.setGeometry(100, 100, 1400, 800)  # x, y, width, height
        
        # Initialize result storage
        self.results = None  # Will store analysis results after solving
        self.current_config = None  # Will store current beam configuration
        
        # Build the user interface
        self.init_ui()
        
    def init_ui(self):
        """
        Initialize and layout all user interface elements.
        
        Creates a split-pane layout:
        - Left panel: Input controls (geometry, loads, supports, profile)
        - Right panel: Visualization and results
        """
        # Create central widget (required for QMainWindow)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        
        # ===== LEFT PANEL: Controls =====
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)  # Constrain width so plots get more space
        
        # --- Beam Geometry Group ---
        geo_group = QGroupBox("Beam Geometry")
        geo_layout = QFormLayout()  # Form layout for label-input pairs
        
        # Beam length input
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(1, 20)  # Min 1m, max 20m
        self.length_spin.setValue(6)  # Default 6m
        self.length_spin.setSingleStep(0.5)  # Increment by 0.5m
        self.length_spin.setSuffix(" m")  # Add unit label
        geo_layout.addRow("Length:", self.length_spin)
        
        geo_group.setLayout(geo_layout)
        left_layout.addWidget(geo_group)
        
        # --- Profile Selection Group ---
        profile_group = QGroupBox("Profile")
        profile_layout = QVBoxLayout()
        
        profile_layout.addWidget(QLabel("METALCON C:"))
        
        # Dropdown to select beam profile
        self.profile_combo = QComboBox()
        for name, props in PROFILES.items():
            # Create descriptive label showing dimensions
            label = f"{name} ({props['A']}×{props['B']}×{props['e']}mm)"
            self.profile_combo.addItem(label, name)  # Display label, store key
        self.profile_combo.setCurrentIndex(2)  # Default to 90CA085
        profile_layout.addWidget(self.profile_combo)
        
        # Label to display weight of selected profile
        self.weight_label = QLabel()
        self.update_weight_label()  # Set initial value
        # Update weight label when profile changes
        self.profile_combo.currentIndexChanged.connect(self.update_weight_label)
        profile_layout.addWidget(self.weight_label)
        
        profile_group.setLayout(profile_layout)
        left_layout.addWidget(profile_group)
        
        # --- Supports Group (3 support points) ---
        supports_group = QGroupBox("Supports (3 points)")
        supports_layout = QFormLayout()
        
        # Support 1 controls
        self.support1_pos = QDoubleSpinBox()
        self.support1_pos.setRange(0, 20)  # Position from 0 to beam length
        self.support1_pos.setValue(0)  # Default at left end
        self.support1_pos.setSingleStep(0.1)
        self.support1_pos.setSuffix(" m")
        supports_layout.addRow("Support 1 Position:", self.support1_pos)
        
        # Support type: pinned (fixed x,y) or roller (fixed y only)
        self.support1_type = QComboBox()
        self.support1_type.addItems(["Pinned", "Roller"])
        supports_layout.addRow("Support 1 Type:", self.support1_type)
        
        # Support 2 controls
        self.support2_pos = QDoubleSpinBox()
        self.support2_pos.setRange(0, 20)
        self.support2_pos.setValue(3)  # Default at middle
        self.support2_pos.setSingleStep(0.1)
        self.support2_pos.setSuffix(" m")
        supports_layout.addRow("Support 2 Position:", self.support2_pos)
        
        self.support2_type = QComboBox()
        self.support2_type.addItems(["Pinned", "Roller"])
        self.support2_type.setCurrentIndex(1)  # Default to roller
        supports_layout.addRow("Support 2 Type:", self.support2_type)
        
        # Support 3 controls
        self.support3_pos = QDoubleSpinBox()
        self.support3_pos.setRange(0, 20)
        self.support3_pos.setValue(6)  # Default at right end
        self.support3_pos.setSingleStep(0.1)
        self.support3_pos.setSuffix(" m")
        supports_layout.addRow("Support 3 Position:", self.support3_pos)
        
        self.support3_type = QComboBox()
        self.support3_type.addItems(["Pinned", "Roller"])
        self.support3_type.setCurrentIndex(1)  # Default to roller
        supports_layout.addRow("Support 3 Type:", self.support3_type)
        
        supports_group.setLayout(supports_layout)
        left_layout.addWidget(supports_group)
        
        # --- Loads Group (2 point loads) ---
        loads_group = QGroupBox("Loads (2 point loads)")
        loads_layout = QFormLayout()
        
        # Load 1 controls
        self.load1_pos = QDoubleSpinBox()
        self.load1_pos.setRange(0, 20)  # Position along beam
        self.load1_pos.setValue(1.5)  # Default position
        self.load1_pos.setSingleStep(0.1)
        self.load1_pos.setSuffix(" m")
        loads_layout.addRow("Load 1 Position:", self.load1_pos)
        
        # Force magnitude (negative = downward, which is typical)
        self.load1_force = QDoubleSpinBox()
        self.load1_force.setRange(-100000, 0)  # Negative for downward
        self.load1_force.setValue(-5000)  # Default 5kN downward
        self.load1_force.setSingleStep(500)
        self.load1_force.setSuffix(" N")
        loads_layout.addRow("Load 1 Force:", self.load1_force)
        
        # Load 2 controls
        self.load2_pos = QDoubleSpinBox()
        self.load2_pos.setRange(0, 20)
        self.load2_pos.setValue(4.5)  # Default position
        self.load2_pos.setSingleStep(0.1)
        self.load2_pos.setSuffix(" m")
        loads_layout.addRow("Load 2 Position:", self.load2_pos)
        
        self.load2_force = QDoubleSpinBox()
        self.load2_force.setRange(-100000, 0)
        self.load2_force.setValue(-3000)  # Default 3kN downward
        self.load2_force.setSingleStep(500)
        self.load2_force.setSuffix(" N")
        loads_layout.addRow("Load 2 Force:", self.load2_force)
        
        loads_group.setLayout(loads_layout)
        left_layout.addWidget(loads_group)
        
        # --- Analyze Button ---
        self.analyze_btn = QPushButton("🔍 Analyze Beam")
        self.analyze_btn.clicked.connect(self.analyze)  # Connect to analysis function
        # Style the button with CSS
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                padding: 12px;
                font-weight: bold;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)
        left_layout.addWidget(self.analyze_btn)
        
        # --- Export Buttons ---
        export_layout = QVBoxLayout()
        
        # PDF Export button (initially disabled until analysis is run)
        self.export_pdf_btn = QPushButton("📄 Export to PDF")
        self.export_pdf_btn.clicked.connect(self.export_pdf)
        self.export_pdf_btn.setEnabled(False)  # Disable until results exist
        self.export_pdf_btn.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #d1d5db;
                color: #6b7280;
            }
        """)
        export_layout.addWidget(self.export_pdf_btn)
        
        # Print button (initially disabled until analysis is run)
        self.print_btn = QPushButton("🖨️ Print")
        self.print_btn.clicked.connect(self.print_report)
        self.print_btn.setEnabled(False)
        self.print_btn.setStyleSheet("""
            QPushButton {
                background-color: #7c3aed;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #6d28d9;
            }
            QPushButton:disabled {
                background-color: #d1d5db;
                color: #6b7280;
            }
        """)
        export_layout.addWidget(self.print_btn)
        
        left_layout.addLayout(export_layout)
        
        # Add stretch to push everything to the top
        left_layout.addStretch()
        
        # ===== RIGHT PANEL: Visualization and Results =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Canvas for displaying matplotlib plots
        self.canvas = MplCanvas(self)
        right_layout.addWidget(self.canvas)
        
        # Text area for displaying numerical results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)  # User cannot edit
        self.results_text.setMaximumHeight(120)  # Limit height
        right_layout.addWidget(self.results_text)
        
        # ===== Combine panels with splitter =====
        # Splitter allows user to resize panels by dragging the divider
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)  # Give more space to right panel
        
        main_layout.addWidget(splitter)
        
    def update_weight_label(self):
        """
        Update the weight label when user selects a different profile.
        
        Displays the linear weight (kg/m) of the selected METALCON profile.
        """
        profile_name = self.profile_combo.currentData()  # Get selected profile key
        weight = PROFILES[profile_name]['weight']  # Look up weight
        self.weight_label.setText(f"Weight: {weight} kg/m")
        
    def analyze(self):
        """
        Execute the beam analysis when user clicks "Analyze Beam" button.
        
        Process:
        1. Gather all input parameters from UI controls
        2. Create BeamSolver instance
        3. Run FEA solver
        4. Store results and configuration
        5. Update visualization
        6. Enable export buttons
        """
        # Gather beam geometry
        length = self.length_spin.value()
        
        # Gather support definitions
        supports = [
            (self.support1_pos.value(), self.support1_type.currentText().lower()),
            (self.support2_pos.value(), self.support2_type.currentText().lower()),
            (self.support3_pos.value(), self.support3_type.currentText().lower()),
        ]
        
        # Gather load definitions
        loads = [
            (self.load1_pos.value(), self.load1_force.value()),
            (self.load2_pos.value(), self.load2_force.value()),
        ]
        
        # Get selected profile information
        profile_name = self.profile_combo.currentData()
        profile = PROFILES[profile_name]
        
        # Create solver instance and run analysis
        solver = BeamSolver(length, supports, loads, profile)
        self.results = solver.solve()  # Store results for later use
        
        # Store current configuration for PDF export
        self.current_config = {
            'length': length,
            'supports': supports,
            'loads': loads,
            'profile_name': profile_name,
            'profile': profile
        }
        
        # Update the plot with new results
        self.plot_results(length, supports, loads)
        
        # Display numerical results in text area
        self.display_results()
        
        # Enable export and print buttons now that we have results
        self.export_pdf_btn.setEnabled(True)
        self.print_btn.setEnabled(True)
        
    def plot_results(self, length, supports, loads):
        """
        Generate and display all beam diagrams on the matplotlib canvas.
        
        Creates 4 subplots:
        1. Beam schematic showing geometry, supports, and loads
        2. Deflection diagram (vertical displacement vs position)
        3. Shear force diagram
        4. Bending moment diagram
        
        Args:
            length (float): Beam length in meters
            supports (list): List of support definitions
            loads (list): List of load definitions
        """
        # Clear any existing plots
        self.canvas.fig.clear()
        
        # Exit if no results to plot
        if not self.results:
            return
        
        # Get x-positions and results from solver
        x = self.results['x']
        
        # Create 4 vertically stacked subplots
        ax1 = self.canvas.fig.add_subplot(4, 1, 1)  # Beam schematic
        ax2 = self.canvas.fig.add_subplot(4, 1, 2)  # Deflection
        ax3 = self.canvas.fig.add_subplot(4, 1, 3)  # Shear
        ax4 = self.canvas.fig.add_subplot(4, 1, 4)  # Moment
        
        # ===== SUBPLOT 1: Beam Schematic =====
        # Draw beam as a horizontal black line
        ax1.plot([0, length], [0, 0], 'k-', linewidth=3, label='Beam')
        
        # Draw support symbols at support locations
        for pos, stype in supports:
            if stype == 'pinned':
                # Triangle pointing up for pinned support
                ax1.plot(pos, 0, '^', markersize=12, color='green', 
                        label='Pinned' if pos == supports[0][0] else '')
            else:
                # Circle for roller support
                ax1.plot(pos, 0, 'o', markersize=10, color='blue', 
                        label='Roller' if pos == supports[1][0] else '')
        
        # Draw load arrows and labels
        for pos, force in loads:
            # Draw downward arrow for load
            ax1.arrow(pos, 0.5, 0, -0.4, head_width=0.1, head_length=0.08, 
                     fc='red', ec='red')
            # Label with force magnitude
            ax1.text(pos, 0.6, f'{force:.0f}N', ha='center', fontsize=9)
        
        # Configure subplot appearance
        ax1.set_xlim(-0.2, length + 0.2)  # Add margins
        ax1.set_ylim(-0.5, 1)
        ax1.set_ylabel('Beam')
        ax1.grid(True, alpha=0.3)  # Light grid
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_title('Beam Configuration', fontweight='bold')
        
        # ===== SUBPLOT 2: Deflection Diagram =====
        # Plot deflection as blue line
        ax2.plot(x, self.results['deflection'], 'b-', linewidth=2)
        # Fill area under curve for better visualization
        ax2.fill_between(x, 0, self.results['deflection'], alpha=0.3)
        ax2.set_ylabel('Deflection (mm)')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Deflection Diagram', fontweight='bold')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Zero reference line
        
        # ===== SUBPLOT 3: Shear Force Diagram =====
        # Plot shear force as green line
        ax3.plot(x, self.results['shear'], 'g-', linewidth=2)
        # Fill area for visualization
        ax3.fill_between(x, 0, self.results['shear'], alpha=0.3, color='green')
        ax3.set_ylabel('Shear (kN)')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Shear Force Diagram', fontweight='bold')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Zero reference
        
        # ===== SUBPLOT 4: Bending Moment Diagram =====
        # Plot moment as red line
        ax4.plot(x, self.results['moment'], 'r-', linewidth=2)
        # Fill area for visualization
        ax4.fill_between(x, 0, self.results['moment'], alpha=0.3, color='red')
        ax4.set_ylabel('Moment (kN⋅m)')
        ax4.set_xlabel('Position (m)')  # X-axis label only on bottom subplot
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Bending Moment Diagram', fontweight='bold')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Zero reference
        
        # Adjust spacing between subplots to prevent overlap
        self.canvas.fig.tight_layout()
        # Redraw the canvas to show updated plots
        self.canvas.draw()
        
    def display_results(self):
        """
        Display numerical analysis results in the text area.
        
        Shows:
        - Support reactions (force at each support)
        - Maximum deflection (largest vertical displacement)
        - Maximum shear force
        - Maximum bending moment
        """
        # Exit if no results to display
        if not self.results:
            return
        
        # Build formatted text string
        text = "=== ANALYSIS RESULTS ===\n\n"
        
        # List all support reactions
        text += "SUPPORT REACTIONS:\n"
        for idx, (pos, reaction) in enumerate(self.results['reactions'], 1):
            text += f"Support {idx} (at {pos:.2f}m): {reaction:.2f} N (↑)\n"
        
        # Display maximum values
        text += f"\nMAX DEFLECTION: {min(self.results['deflection']):.3f} mm\n"
        text += f"MAX SHEAR: {max(abs(self.results['shear'])):.2f} kN\n"
        text += f"MAX MOMENT: {max(abs(self.results['moment'])):.2f} kN⋅m\n"
        
        # Update the text widget
        self.results_text.setText(text)
    
    def export_pdf(self):
        """
        Export a complete analysis report to PDF file.
        
        Creates a professional one-page report including:
        - Header with date, profile info, and beam geometry
        - All 4 diagrams (schematic, deflection, shear, moment)
        - Numerical results summary
        
        User selects save location via file dialog.
        """
        # Exit if no results to export
        if not self.results:
            return
        
        # Open file save dialog
        # Default filename includes timestamp
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", 
            f"beam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF Files (*.pdf)"
        )
        
        # Exit if user cancelled
        if not filename:
            return
        
        try:
            # Create a new figure specifically for PDF (letter size paper)
            fig = plt.figure(figsize=(8.5, 11))  # 8.5" x 11" = US Letter
            
            # Add main title at top
            fig.suptitle('METALCON Beam Analysis Report', fontsize=16, fontweight='bold')
            
            # Add metadata text at top of page
            info_text = f"""
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Profile: {self.current_config['profile_name']} ({self.current_config['profile']['A']}×{self.current_config['profile']['B']}×{self.current_config['profile']['e']}mm)
Beam Length: {self.current_config['length']:.2f} m
            """
            fig.text(0.1, 0.95, info_text, fontsize=10, verticalalignment='top', family='monospace')
            
            # Create 5 subplots for PDF
            ax1 = fig.add_subplot(5, 1, 1)  # Beam schematic
            ax2 = fig.add_subplot(5, 1, 2)  # Deflection
            ax3 = fig.add_subplot(5, 1, 3)  # Shear
            ax4 = fig.add_subplot(5, 1, 4)  # Moment
            ax5 = fig.add_subplot(5, 1, 5)  # Results text
            
            # Get configuration for plotting
            length = self.current_config['length']
            supports = self.current_config['supports']
            loads = self.current_config['loads']
            x = self.results['x']
            
            # ===== Plot 1: Beam Schematic =====
            ax1.plot([0, length], [0, 0], 'k-', linewidth=3)
            # Draw supports
            for pos, stype in supports:
                if stype == 'pinned':
                    ax1.plot(pos, 0, '^', markersize=10, color='green')
                else:
                    ax1.plot(pos, 0, 'o', markersize=8, color='blue')
            # Draw loads
            for pos, force in loads:
                ax1.arrow(pos, 0.3, 0, -0.25, head_width=0.05, head_length=0.04, 
                         fc='red', ec='red')
                ax1.text(pos, 0.35, f'{force:.0f}N', ha='center', fontsize=8)
            ax1.set_xlim(-0.2, length + 0.2)
            ax1.set_ylim(-0.3, 0.5)
            ax1.set_title('Beam Configuration', fontweight='bold', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylabel('Beam')
            
            # ===== Plot 2: Deflection =====
            ax2.plot(x, self.results['deflection'], 'b-', linewidth=1.5)
            ax2.fill_between(x, 0, self.results['deflection'], alpha=0.3)
            ax2.set_ylabel('Deflection (mm)', fontsize=9)
            ax2.set_title('Deflection Diagram', fontweight='bold', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            # ===== Plot 3: Shear Force =====
            ax3.plot(x, self.results['shear'], 'g-', linewidth=1.5)
            ax3.fill_between(x, 0, self.results['shear'], alpha=0.3, color='green')
            ax3.set_ylabel('Shear (kN)', fontsize=9)
            ax3.set_title('Shear Force Diagram', fontweight='bold', fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            # ===== Plot 4: Bending Moment =====
            ax4.plot(x, self.results['moment'], 'r-', linewidth=1.5)
            ax4.fill_between(x, 0, self.results['moment'], alpha=0.3, color='red')
            ax4.set_ylabel('Moment (kN⋅m)', fontsize=9)
            ax4.set_title('Bending Moment Diagram', fontweight='bold', fontsize=10)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            # ===== Plot 5: Results Summary Text =====
            ax5.axis('off')  # Hide axes for text display
            # Build results text
            results_text = "ANALYSIS RESULTS\n" + "="*50 + "\n\n"
            results_text += "Support Reactions:\n"
            for idx, (pos, reaction) in enumerate(self.results['reactions'], 1):
                results_text += f"  Support {idx} (at {pos:.2f}m): {reaction:.2f} N ↑\n"
            results_text += f"\nMax Deflection: {min(self.results['deflection']):.3f} mm\n"
            results_text += f"Max Shear: {max(abs(self.results['shear'])):.2f} kN\n"
            results_text += f"Max Moment: {max(abs(self.results['moment'])):.2f} kN⋅m"
            
            # Display text in a box
            ax5.text(0.1, 0.5, results_text, fontsize=9, verticalalignment='center', 
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Adjust layout to fit everything nicely
            plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space for title
            
            # Save figure to PDF with high resolution
            plt.savefig(filename, format='pdf', dpi=300)
            plt.close(fig)  # Close to free memory
            
            # Show success message
            QMessageBox.information(self, "Success", f"Report exported to:\n{filename}")
            
        except Exception as e:
            # Show error message if something goes wrong
            QMessageBox.critical(self, "Error", f"Failed to export PDF:\n{str(e)}")
    
    def print_report(self):
        """
        Print the analysis report.
        
        Note: Direct printing from matplotlib can be tricky, so this function
        currently recommends using PDF export instead for best results.
        A full print implementation would require converting the figure to
        a printable format (e.g., QPainter).
        """
        # Exit if no results to print
        if not self.results:
            return
        
        # Create printer object
        printer = QPrinter(QPrinter.HighResolution)
        # Open print dialog for user to select printer and settings
        dialog = QPrintDialog(printer, self)
        
        if dialog.exec_() == QPrintDialog.Accepted:
            # Printing directly from matplotlib is complex
            # For now, recommend using PDF export
            # A full implementation would use QPainter to render the figure
            try:
                # Save temporary PDF for printing
                self.canvas.fig.savefig('temp_print.pdf', format='pdf')
                
                # Show message recommending PDF export
                QMessageBox.information(self, "Print", 
                    "For best results, use 'Export to PDF' and print the PDF file.")
            except Exception as e:
                QMessageBox.warning(self, "Print", 
                    "Print function is limited. Please use 'Export to PDF' instead.")

def main():
    """
    Main entry point for the application.
    
    Creates the Qt application, initializes the main window, and starts
    the event loop.
    """
    # Create Qt application instance
    # sys.argv allows passing command-line arguments if needed
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start Qt event loop and exit when closed
    sys.exit(app.exec_())

# Standard Python idiom: only run if this file is executed directly
# (not if it's imported as a module)
if __name__ == '__main__':
    main()