"""
METALCON Calculadora Vigas Simples - Version 1.0
Aplicacion de escritorio para analisis estructural de vigas simples con multiples apoyos y cargas variables.
GNU GPLv3 - Felipe Carvajal Brown
"""

import sys
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QSpinBox, 
                             QPushButton, QTextEdit, QGroupBox, QDoubleSpinBox,
                             QSplitter, QFormLayout, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# METALCON C profiles from catalog
PROFILES = {
    '40CA085': {'height': 40, 'width': 38, 'thickness': 0.85, 'weight': 0.83, 'moment_of_inertia': 2.1e4},
    '60CA085': {'height': 60, 'width': 38, 'thickness': 0.85, 'weight': 0.96, 'moment_of_inertia': 4.8e4},
    '90CA085': {'height': 90, 'width': 38, 'thickness': 0.85, 'weight': 1.23, 'moment_of_inertia': 1.1e5},
    '100CA085': {'height': 100, 'width': 40, 'thickness': 0.85, 'weight': 1.32, 'moment_of_inertia': 1.4e5},
    '150CA085': {'height': 150, 'width': 40, 'thickness': 0.85, 'weight': 1.64, 'moment_of_inertia': 3.2e5},
    '90CA10': {'height': 90, 'width': 38, 'thickness': 1.0, 'weight': 1.44, 'moment_of_inertia': 1.3e5},
    '150CA10': {'height': 150, 'width': 40, 'thickness': 1.0, 'weight': 1.94, 'moment_of_inertia': 3.8e5},
    '150CA16': {'height': 150, 'width': 40, 'thickness': 1.6, 'weight': 3.06, 'moment_of_inertia': 5.9e5},
}

# METALCON U profiles from catalog
U_PROFILES = {
    '42C085': {'height': 42, 'width': 25, 'thickness': 0.85, 'weight': 0.58, 'moment_of_inertia': 1},
    '62C085_3': {'height': 62, 'width': 25, 'thickness': 0.85, 'weight': 0.72, 'moment_of_inertia': 1},
    '62C085_6': {'height': 62, 'width': 25, 'thickness': 0.85, 'weight': 0.72, 'moment_of_inertia': 1},
    '92C085_3': {'height': 92, 'width': 30, 'thickness': 0.85, 'weight': 1.00, 'moment_of_inertia': 1},
    '92C085_3b': {'height': 92, 'width': 30, 'thickness': 0.85, 'weight': 1.00, 'moment_of_inertia': 1},
    '92C10': {'height': 92, 'width': 30, 'thickness': 1.00, 'weight': 1.17, 'moment_of_inertia': 1},
    '103C085': {'height': 103, 'width': 30, 'thickness': 0.85, 'weight': 1.06, 'moment_of_inertia': 1},
    '103C10': {'height': 103, 'width': 30, 'thickness': 1.00, 'weight': 1.25, 'moment_of_inertia': 1},
    '153C10': {'height': 153, 'width': 30, 'thickness': 1.00, 'weight': 1.65, 'moment_of_inertia': 1},
    '203C10': {'height': 203, 'width': 30, 'thickness': 1.00, 'weight': 2.04, 'moment_of_inertia': 1},
}

class BeamSolver:
    """FEA solver for simple beam with multiple supports"""
    
    def __init__(self, length, supports, loads, profile, E=200000):
        """
        length: beam length in meters
        supports: list of (position, type) where type is 'pinned' or 'roller'
        loads: list of (position, force) tuples (force in N, negative = downward)
        profile: METALCON profile dict
        E: Young's modulus in MPa
        """
        self.L = length
        self.supports = sorted(supports, key=lambda x: x[0])  # Sort by position
        self.loads = loads
        self.profile = profile
        self.E = E * 1e6  # Convert to Pa
        
        # Moment of inertia (mm^4 to m^4)
        self.I = profile['moment_of_inertia'] / 1e12
        
        # Number of elements for FEA
        self.n_elements = 50
        
    def solve(self):
        """Solve beam using finite element method"""
        n_elem = self.n_elements
        n_nodes = n_elem + 1
        elem_length = self.L / n_elem
        
        # Node positions
        x_nodes = np.linspace(0, self.L, n_nodes)
        
        # DOFs: 2 per node (vertical displacement w, rotation theta)
        n_dof = n_nodes * 2
        
        # Global stiffness matrix and force vector
        K = np.zeros((n_dof, n_dof))
        F = np.zeros(n_dof)
        
        # Assemble stiffness matrix (Euler-Bernoulli beam elements)
        for i in range(n_elem):
            Le = elem_length
            EI = self.E * self.I
            
            # Element stiffness matrix
            ke = (EI / Le**3) * np.array([
                [12, 6*Le, -12, 6*Le],
                [6*Le, 4*Le**2, -6*Le, 2*Le**2],
                [-12, -6*Le, 12, -6*Le],
                [6*Le, 2*Le**2, -6*Le, 4*Le**2]
            ])
            
            # Global DOFs for this element
            dofs = [i*2, i*2+1, (i+1)*2, (i+1)*2+1]
            
            # Add to global matrix
            for ii in range(4):
                for jj in range(4):
                    K[dofs[ii], dofs[jj]] += ke[ii, jj]
        
        # Apply point loads
        for pos, force in self.loads:
            # Find closest node
            node_idx = int(round(pos / self.L * n_elem))
            if 0 <= node_idx < n_nodes:
                F[node_idx * 2] += force  # Force in w direction
        
        # Apply boundary conditions
        fixed_dofs = []
        for pos, support_type in self.supports:
            node_idx = int(round(pos / self.L * n_elem))
            if 0 <= node_idx < n_nodes:
                # Fix vertical displacement for all supports
                fixed_dofs.append(node_idx * 2)
                # Fix rotation only for pinned supports (not roller)
                if support_type == 'pinned':
                    fixed_dofs.append(node_idx * 2 + 1)
        
        # Modify system for boundary conditions
        K_mod = K.copy()
        F_mod = F.copy()
        
        for dof in fixed_dofs:
            K_mod[dof, :] = 0
            K_mod[:, dof] = 0
            K_mod[dof, dof] = 1
            F_mod[dof] = 0
        
        # Solve for displacements
        try:
            U = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            U = np.zeros(n_dof)
        
        # Extract displacements and rotations
        w = U[::2]  # Vertical displacements
        theta = U[1::2]  # Rotations
        
        # Calculate reactions at supports
        reactions = []
        for pos, support_type in self.supports:
            node_idx = int(round(pos / self.L * n_elem))
            if 0 <= node_idx < n_nodes:
                # Reaction = K * u - F
                reaction = 0
                for j in range(n_dof):
                    reaction += K[node_idx * 2, j] * U[j]
                reaction -= F[node_idx * 2]
                reactions.append((pos, -reaction))  # Negative for upward reaction
        
        # Calculate shear and moment along beam
        shear = np.zeros(n_nodes)
        moment = np.zeros(n_nodes)
        
        for i in range(n_nodes):
            x = x_nodes[i]
            
            # Add reactions (left of point)
            for pos, reaction in reactions:
                if pos < x:
                    shear[i] += reaction
            
            # Subtract loads (left of point)
            for pos, force in self.loads:
                if pos < x:
                    shear[i] += force  # force is already negative for downward
            
            # Calculate moment
            for pos, reaction in reactions:
                if pos < x:
                    moment[i] += reaction * (x - pos)
            
            for pos, force in self.loads:
                if pos < x:
                    moment[i] += force * (x - pos)
        
        return {
            'x': x_nodes,
            'deflection': w * 1000,  # Convert to mm
            'rotation': theta,
            'shear': shear / 1000,  # Convert to kN
            'moment': moment / 1000,  # Convert to kN⋅m
            'reactions': reactions
        }

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for plotting"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), facecolor='white')
        super().__init__(self.fig)
        self.setParent(parent)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("METALCON Calculadora Vigas Simples")
        self.setGeometry(100, 100, 1400, 800)
        
        self.results = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)
        
        # Beam geometry group
        geo_group = QGroupBox("Perfil de Viga")
        geo_layout = QFormLayout()
        
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(1, 20)
        self.length_spin.setValue(6)
        self.length_spin.setSingleStep(0.5)
        self.length_spin.setSuffix(" m")
        geo_layout.addRow("Longitud:", self.length_spin)
        
        geo_group.setLayout(geo_layout)
        left_layout.addWidget(geo_group)
        
        # Profile group
        profile_group = QGroupBox("Perfil")
        profile_layout = QVBoxLayout()
        
        profile_layout.addWidget(QLabel("METALCON C:"))
        self.profile_combo = QComboBox()
        for name, props in PROFILES.items():
            label = f"{name} ({props['height']}×{props['width']}×{props['thickness']}mm)"
            self.profile_combo.addItem(label, name)
        self.profile_combo.setCurrentIndex(2)  # 90CA085
        profile_layout.addWidget(self.profile_combo)
        
        self.weight_label = QLabel()
        self.update_weight_label()
        self.profile_combo.currentIndexChanged.connect(self.update_weight_label)
        profile_layout.addWidget(self.weight_label)
        
        profile_group.setLayout(profile_layout)
        left_layout.addWidget(profile_group)
        
        # Supports group
        supports_group = QGroupBox("Soportes (3 apoyos)")
        supports_layout = QFormLayout()
        
        self.support1_pos = QDoubleSpinBox()
        self.support1_pos.setRange(0, 20)
        self.support1_pos.setValue(0)
        self.support1_pos.setSingleStep(0.1)
        self.support1_pos.setSuffix(" m")
        supports_layout.addRow("Posición Soporte 1:", self.support1_pos)
        
        self.support1_type = QComboBox()
        self.support1_type.addItems(["Pinned", "Roller"])
        supports_layout.addRow("Tipo Soporte 1:", self.support1_type)
        
        self.support2_pos = QDoubleSpinBox()
        self.support2_pos.setRange(0, 20)
        self.support2_pos.setValue(3)
        self.support2_pos.setSingleStep(0.1)
        self.support2_pos.setSuffix(" m")
        supports_layout.addRow("Posición Soporte 2:", self.support2_pos)
        
        self.support2_type = QComboBox()
        self.support2_type.addItems(["Pinned", "Roller"])
        self.support2_type.setCurrentIndex(1)
        supports_layout.addRow("Tipo Soporte 2:", self.support2_type)
        
        self.support3_pos = QDoubleSpinBox()
        self.support3_pos.setRange(0, 20)
        self.support3_pos.setValue(6)
        self.support3_pos.setSingleStep(0.1)
        self.support3_pos.setSuffix(" m")
        supports_layout.addRow("Posición Soporte 3:", self.support3_pos)
        
        self.support3_type = QComboBox()
        self.support3_type.addItems(["Pinned", "Roller"])
        self.support3_type.setCurrentIndex(1)
        supports_layout.addRow("Tipo Soporte 3:", self.support3_type)
        
        supports_group.setLayout(supports_layout)
        left_layout.addWidget(supports_group)
        
        # Loads group
        loads_group = QGroupBox("Cargas (2 cargas puntuales)")
        loads_layout = QFormLayout()
        
        self.load1_pos = QDoubleSpinBox()
        self.load1_pos.setRange(0, 20)
        self.load1_pos.setValue(1.5)
        self.load1_pos.setSingleStep(0.1)
        self.load1_pos.setSuffix(" m")
        loads_layout.addRow("Posición Carga 1:", self.load1_pos)
        
        self.load1_force = QDoubleSpinBox()
        self.load1_force.setRange(-100000, 0)
        self.load1_force.setValue(-5000)
        self.load1_force.setSingleStep(500)
        self.load1_force.setSuffix(" N")
        loads_layout.addRow("Fuerza Carga 1:", self.load1_force)
        
        self.load2_pos = QDoubleSpinBox()
        self.load2_pos.setRange(0, 20)
        self.load2_pos.setValue(4.5)
        self.load2_pos.setSingleStep(0.1)
        self.load2_pos.setSuffix(" m")
        loads_layout.addRow("Posición Carga 2:", self.load2_pos)
        
        self.load2_force = QDoubleSpinBox()
        self.load2_force.setRange(-100000, 0)
        self.load2_force.setValue(-3000)
        self.load2_force.setSingleStep(500)
        self.load2_force.setSuffix(" N")
        loads_layout.addRow("Fuerza Carga 2:", self.load2_force)
        
        loads_group.setLayout(loads_layout)
        left_layout.addWidget(loads_group)
        
        # Analyze button
        self.analyze_btn = QPushButton("🔍 Analizar Viga")
        self.analyze_btn.clicked.connect(self.analyze)
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
        
        # Export buttons
        export_layout = QVBoxLayout()
        
        self.export_pdf_btn = QPushButton("📄 Exportar a PDF")
        self.export_pdf_btn.clicked.connect(self.export_pdf)
        self.export_pdf_btn.setEnabled(False)
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
        
        self.print_btn = QPushButton("🖨️ Imprimir")
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
        
        left_layout.addStretch()
        
        # Right panel - Visualization and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Canvas for plots
        self.canvas = MplCanvas(self)
        right_layout.addWidget(self.canvas)
        
        # Results text
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(120)
        right_layout.addWidget(self.results_text)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
    def update_weight_label(self):
        """Update weight label when profile changes"""
        profile_name = self.profile_combo.currentData()
        weight = PROFILES[profile_name]['weight']
        self.weight_label.setText(f"Peso: {weight} kg/m")
        
    def analyze(self):
        """Run beam analysis"""
        # Get parameters
        length = self.length_spin.value()
        
        supports = [
            (self.support1_pos.value(), self.support1_type.currentText().lower()),
            (self.support2_pos.value(), self.support2_type.currentText().lower()),
            (self.support3_pos.value(), self.support3_type.currentText().lower()),
        ]
        
        loads = [
            (self.load1_pos.value(), self.load1_force.value()),
            (self.load2_pos.value(), self.load2_force.value()),
        ]
        
        profile_name = self.profile_combo.currentData()
        profile = PROFILES[profile_name]
        
        # Solve
        solver = BeamSolver(length, supports, loads, profile)
        self.results = solver.solve()
        
        # Store current config for export
        self.current_config = {
            'length': length,
            'supports': supports,
            'loads': loads,
            'profile_name': profile_name,
            'profile': profile
        }
        
        # Plot results
        self.plot_results(length, supports, loads)
        
        # Display text results
        self.display_results()
        
        # Enable export buttons
        self.export_pdf_btn.setEnabled(True)
        self.print_btn.setEnabled(True)
        
    def plot_results(self, length, supports, loads):
        """Plot beam diagrams"""
        self.canvas.fig.clear()
        
        if not self.results:
            return
        
        x = self.results['x']
        
        # Create 4 subplots
        ax1 = self.canvas.fig.add_subplot(4, 1, 1)
        ax2 = self.canvas.fig.add_subplot(4, 1, 2)
        ax3 = self.canvas.fig.add_subplot(4, 1, 3)
        ax4 = self.canvas.fig.add_subplot(4, 1, 4)
        
        # 1. Beam schematic
        ax1.plot([0, length], [0, 0], 'k-', linewidth=3, label='Viga')
        
        # Draw supports
        for pos, stype in supports:
            if stype == 'pinned':
                ax1.plot(pos, 0, '^', markersize=12, color='green', label='Apoyo empotrado' if pos == supports[0][0] else '')
            else:
                ax1.plot(pos, 0, 'o', markersize=10, color='blue', label='Rodillo' if pos == supports[1][0] else '')
        
        # Draw loads
        for pos, force in loads:
            ax1.arrow(pos, 0.5, 0, -0.4, head_width=0.1, head_length=0.08, fc='red', ec='red')
            ax1.text(pos, 0.6, f'{force:.0f}N', ha='center', fontsize=9)
        
        ax1.set_xlim(-0.2, length + 0.2)
        ax1.set_ylim(-0.5, 1)
        ax1.set_ylabel('Viga')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_title('Configuración de Viga', fontweight='bold')
        
        # 2. Deflection
        ax2.plot(x, self.results['deflection'], 'b-', linewidth=2)
        ax2.fill_between(x, 0, self.results['deflection'], alpha=0.3)
        ax2.set_ylabel('Desviación (mm)')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Diagrama de Desviación', fontweight='bold')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 3. Shear
        ax3.plot(x, self.results['shear'], 'g-', linewidth=2)
        ax3.fill_between(x, 0, self.results['shear'], alpha=0.3, color='green')
        ax3.set_ylabel('Shear (kN)')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Diagrama de Fuerza Cortante', fontweight='bold')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 4. Moment
        ax4.plot(x, self.results['moment'], 'r-', linewidth=2)
        ax4.fill_between(x, 0, self.results['moment'], alpha=0.3, color='red')
        ax4.set_ylabel('Momento (kN⋅m)')
        ax4.set_xlabel('Posición (m)')
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Diagrama de Momento Flector', fontweight='bold')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
    def display_results(self):
        """Display numerical results"""
        if not self.results:
            return
        
        text = "=== RESULTADOS DEL ANÁLISIS ===\n\n"
        
        text += "REACCIONES EN LOS APOYOS:\n"
        for idx, (pos, reaction) in enumerate(self.results['reactions'], 1):
            text += f"Apoyo {idx} (en {pos:.2f}m): {reaction:.2f} N (↑)\n"
        
        text += f"\nMAX DESVIACIÓN: {min(self.results['deflection']):.3f} mm\n"
        text += f"MAX shear: {max(abs(self.results['shear'])):.2f} kN\n"
        text += f"MAX MOMENTO: {max(abs(self.results['moment'])):.2f} kN⋅m\n"
        
        self.results_text.setText(text)
    
    def export_pdf(self):
        """Export report to PDF"""
        if not self.results:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Guardar Informe PDF", 
            f"analisis_viga_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "Archivos PDF (*.pdf)"
        )
        
        if not filename:
            return
        
        try:
            # Save the canvas figure directly to PDF
            self.canvas.fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Success", f"Report exported to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export PDF:\n{str(e)}")
    def export_pdf(self):
        """Export report to PDF"""
        if not self.results:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Guardar Informe PDF", 
            f"analisis_viga_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "Archivos PDF (*.pdf)"
        )
        
        if not filename:
            return
        
        try:
            # Save the canvas figure directly to PDF
            self.canvas.fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Success", f"Report exported to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export PDF:\n{str(e)}")
        """Export report to PDF"""
        if not self.results:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Guardar Informe PDF", 
            f"analisis_viga_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "Archivos PDF (*.pdf)"
        )
        
        if not filename:
            return
        
        try:
            with PdfPages(filename) as pdf:
                # Save the canvas figure
                pdf.savefig(self.canvas.fig, bbox_inches='tight')
                
                # Add metadata
                d = pdf.infodict()
                d['Title'] = 'METALCON Reporte de Análisis de Viga'
                d['Author'] = 'METALCON Calculadora Vigas Simples'
                d['Subject'] = 'Análisis estructural de vigas simples'
                d['CreationDate'] = datetime.now()
            
            QMessageBox.information(self, "Success", f"Report exported to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export PDF:\n{str(e)}")

    def print_report(self):
        """Print the report"""
        if not self.results:
            return
        
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)
        
        if dialog.exec_() == QPrintDialog.Accepted:
            # Use the canvas figure for printing
            self.canvas.fig.savefig('temp_print.pdf', format='pdf')
            
            # Simple print using Qt
            try:
                painter = self.canvas.fig.canvas
                # This would need more sophisticated implementation
                QMessageBox.information(self, "Print", 
                    "For best results, use 'Export to PDF' and print the PDF file.")
            except Exception as e:
                QMessageBox.warning(self, "Print", 
                    "Print function is limited. Please use 'Export to PDF' instead.")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()