"""
PyQt6 GUI for Sales Forecasting & On-Demand Predictions
"""
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QTextEdit, 
    QStackedWidget, QTableWidget, QTableWidgetItem, QDateEdit,
    QMessageBox, QSplitter, QFrame, QScrollArea, QCheckBox
)
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QColor
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import re

# Import your utility functions
from utils.models_functions import (
    forecast_sales, forecast_demand, 
    train_randomforest_demand_forecasting, train_randomforest_sales_forecasting
)
from utils.preprocessing import preprocess_ungrouped, grouping_data, feature_engineering, time_series_features, handle_category, split_numeric_categorical


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


def style_dark_axis(ax):
    """Apply dark theme styling to matplotlib axis"""
    ax.set_facecolor('#2a2a2a')
    ax.tick_params(colors='#808080', labelsize=9)
    ax.grid(True, alpha=0.2, color='#404040', linestyle='-', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color('#404040')
    ax.xaxis.label.set_color('#e0e0e0')
    ax.yaxis.label.set_color('#e0e0e0')
    ax.title.set_color('#ffffff')
    ax.title.set_fontsize(12)


def show_message_box(parent, icon, title, text):
    """Show a styled message box with visible buttons"""
    msg = QMessageBox(parent)
    msg.setIcon(icon)
    msg.setWindowTitle(title)
    msg.setText(text)
    
    # Style the message box and buttons
    msg.setStyleSheet("""
        QMessageBox {
            background-color: #2a2a2a;
        }
        QMessageBox QLabel {
            color: #e0e0e0;
            padding: 10px;
            min-width: 300px;
        }
        QPushButton {
            background-color: #4a90e2;
            color: #ffffff;
            padding: 10px 25px;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            font-size: 13px;
            min-width: 80px;
            min-height: 30px;
        }
        QPushButton:hover {
            background-color: #357abd;
        }
        QPushButton:pressed {
            background-color: #2868a8;
        }
    """)
    
    msg.exec()


class MatplotlibWidget(QWidget):
    """Widget to embed matplotlib figures with interactive controls"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 6), facecolor='#1a1a1a')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1a1a1a;")
        
        # Add navigation toolbar for interactive features (zoom, pan, save)
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2a2a2a;
                border: none;
                padding: 5px;
            }
            QToolButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                padding: 5px;
                margin: 2px;
            }
            QToolButton:hover {
                background-color: #4a4a4a;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def clear(self):
        self.figure.clear()
        self.figure.patch.set_facecolor('#1a1a1a')
        self.canvas.draw()


class TrainingThread(QThread):
    """Background thread for model training"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)
    
    def __init__(self, df_demand, df_series, df_marketing):
        super().__init__()
        self.df_demand = df_demand
        self.df_series = df_series
        self.df_marketing = df_marketing
    
    def run(self):
        try:
            self.progress.emit("Training demand forecasting models with Random Forest...")
            train_randomforest_demand_forecasting(self.df_demand)
            
            self.progress.emit("Training sales forecasting models with Random Forest...")
            train_randomforest_sales_forecasting(self.df_series, self.df_marketing)
            
            self.finished.emit(True, "✓ Random Forest models trained successfully!")
        except Exception as e:
            self.finished.emit(False, f"Training failed: {str(e)}")


class SalesForecastingPage(QWidget):
    """Sales Forecasting Page"""
    def __init__(self):
        super().__init__()
        self.marketing_data = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Sales Forecasting")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # File upload section
        file_section = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("""
            QLabel {
                color: #808080;
                padding: 10px;
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
        """)
        upload_btn = QPushButton("Upload Marketing Budget CSV")
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #e0e0e0;
                padding: 10px 20px;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #333333;
                border: 1px solid #4a90e2;
            }
        """)
        upload_btn.clicked.connect(self.upload_file)
        file_section.addWidget(upload_btn)
        file_section.addWidget(self.file_label, 1)
        file_section.addStretch()
        layout.addLayout(file_section)
        
        # Date info
        self.date_info = QLabel("")
        self.date_info.setStyleSheet("QLabel { color: #808080; padding: 5px; }")
        layout.addWidget(self.date_info)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Generate button
        generate_btn = QPushButton("Generate Forecast")
        generate_btn.clicked.connect(self.generate_forecast)
        generate_btn.setMinimumHeight(45)
        generate_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00b894, stop:1 #00cec9);
                color: white;
                padding: 12px 24px;
                font-size: 15px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #019875, stop:1 #00b5a8);
            }
            QPushButton:pressed {
                background: #00a085;
            }
        """)
        
        # Export button
        self.export_sales_btn = QPushButton("Export Results")
        self.export_sales_btn.clicked.connect(self.export_sales_forecast)
        self.export_sales_btn.setMinimumHeight(45)
        self.export_sales_btn.setEnabled(False)
        self.export_sales_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #808080;
                padding: 12px 24px;
                font-size: 15px;
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
            }
            QPushButton:enabled {
                background-color: #34495e;
                color: #e0e0e0;
            }
            QPushButton:enabled:hover {
                background-color: #415364;
            }
        """)
        
        buttons_layout.addWidget(generate_btn)
        buttons_layout.addWidget(self.export_sales_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Chart
        self.chart = MatplotlibWidget()
        layout.addWidget(self.chart)
        
        # Metrics
        metrics_layout = QHBoxLayout()
        self.avg_metric = QLabel("Average Daily Sales: -")
        self.total_metric = QLabel("Total Sales: -")
        self.peak_metric = QLabel("Peak Sales: -")
        
        for i, metric in enumerate([self.avg_metric, self.total_metric, self.peak_metric]):
            colors = ['#1a2332', '#2a2416', '#1a2e26']
            borders = ['#4a90e2', '#f39c12', '#27ae60']
            metric.setStyleSheet(f"""
                QLabel {{
                    background-color: {colors[i]};
                    border-left: 4px solid {borders[i]};
                    padding: 20px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 500;
                    color: #e0e0e0;
                }}
            """)
            metrics_layout.addWidget(metric)
        
        layout.addLayout(metrics_layout)
        
        # Quick Tips Panel
        tips_panel = QFrame()
        tips_panel.setStyleSheet("""
            QFrame {
                background-color: #232832;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        tips_layout = QVBoxLayout()
        
        tips_title = QLabel("Quick Tips")
        tips_title.setStyleSheet("QLabel { color: #4a90e2; font-size: 14px; font-weight: bold; }")
        tips_layout.addWidget(tips_title)
        
        tips_text = QLabel(
            "• Upload marketing budget CSV with 'dates' and 'marketing budget' columns\n"
            "• Ensure dates are in YYYY-MM-DD format\n"
            "• Higher marketing budgets typically lead to increased sales\n"
            "• Export results to save forecast data and charts for reports"
        )
        tips_text.setStyleSheet("QLabel { color: #c0c0c0; font-size: 12px; line-height: 1.6; }")
        tips_text.setWordWrap(True)
        tips_layout.addWidget(tips_text)
        
        tips_panel.setLayout(tips_layout)
        layout.addWidget(tips_panel)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.marketing_data = pd.read_csv(file_path)
                self.file_label.setText(f"Loaded: {os.path.basename(file_path)}")
                
                # Show date range
                start_date = self.marketing_data['dates'].min()
                end_date = self.marketing_data['dates'].max()
                self.date_info.setText(f"Start Date: {start_date} | End Date: {end_date}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    

    
    def generate_forecast(self):
        if self.marketing_data is None:
            QMessageBox.warning(self, "Warning", "Please upload a marketing budget file first!")
            return
        
        try:
            # Convert to string format if it's a Timestamp
            start_date = self.marketing_data['dates'].min()
            if hasattr(start_date, 'strftime'):
                start_date = start_date.strftime('%Y-%m-%d')
            elif isinstance(start_date, str):
                # Already a string, ensure proper format
                start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            
            days = len(self.marketing_data['dates'])
            marketing_budget = self.marketing_data['marketing budget'].tolist()
            
            forecast_df = forecast_sales(start_date, days, marketing_budget)
            
            # Plot
            self.chart.clear()
            ax = self.chart.figure.add_subplot(111, facecolor='#2a2a2a')
            ax.plot(forecast_df.index, forecast_df['Forecasted Subtotal'], color='#4a90e2', linewidth=2)
            ax.set_xlabel('Days', color='#e0e0e0')
            ax.set_ylabel('Sales ($)', color='#e0e0e0')
            ax.set_title('Sales Forecast', color='#ffffff', fontsize=14, pad=15)
            ax.tick_params(colors='#808080')
            ax.grid(True, alpha=0.2, color='#404040')
            ax.spines['bottom'].set_color('#404040')
            ax.spines['top'].set_color('#404040')
            ax.spines['left'].set_color('#404040')
            ax.spines['right'].set_color('#404040')
            self.chart.canvas.draw()
            
            # Update metrics
            avg_sales = forecast_df['Forecasted Subtotal'].mean()
            total_sales = forecast_df['Forecasted Subtotal'].sum()
            peak_sales = forecast_df['Forecasted Subtotal'].max()
            
            self.avg_metric.setText(f"Average Daily Sales: ${avg_sales:,.2f}")
            self.total_metric.setText(f"Total Sales: ${total_sales:,.2f}")
            self.peak_metric.setText(f"Peak Sales: ${peak_sales:,.2f}")
            
            # Store forecast for export
            self.forecast_data = forecast_df
            self.export_sales_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating forecast: {str(e)}")
    
    def export_sales_forecast(self):
        if not hasattr(self, 'forecast_data'):
            return
        
        try:
            # Export CSV
            file_path, _ = QFileDialog.getSaveFileName(self, "Export Sales Forecast", "sales_forecast.csv", "CSV Files (*.csv)")
            if file_path:
                self.forecast_data.to_csv(file_path)
                
                # Also save chart
                chart_path = file_path.replace('.csv', '.png')
                self.chart.figure.savefig(chart_path, dpi=300, facecolor='#1a1a1a', edgecolor='none')
                
                QMessageBox.information(self, "Success", f"Exported to:\n{file_path}\n{chart_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting: {str(e)}")


class DemandProductPage(QWidget):
    """Demand Product Analysis Page"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Demand Product Analysis")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Product selection
        product_layout = QHBoxLayout()
        product_layout.addWidget(QLabel("Select Product Category:"))
        self.product_combo = QComboBox()
        self.product_combo.addItems([
            'hand towel', 'towel', 'face towel', 'bath mat', 'mattress protector',
            'mattress topper', 'bathrobe', 'pillow', 'coverlet', 'bed sheet',
            'fitted sheet', 'kitchen towel', 'cushion', 'duvet', 'blanket',
            'beach towel', 'set', 'apron', 'slipper', 'other'
        ])
        product_layout.addWidget(self.product_combo)
        product_layout.addStretch()
        layout.addLayout(product_layout)
        
        # Date selection
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Start Date:"))
        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate(2025, 1, 17))
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setMinimumDate(QDate(2025, 1, 17))
        date_layout.addWidget(self.date_edit)
        date_layout.addStretch()
        layout.addLayout(date_layout)
        
        # Process button
        process_btn = QPushButton("Analyze Demand")
        process_btn.clicked.connect(self.process_demand)
        process_btn.setMinimumHeight(45)
        process_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6c5ce7, stop:1 #a29bfe);
                color: white;
                padding: 12px 24px;
                font-size: 15px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5f3dc4, stop:1 #9189f5);
            }
        """)
        layout.addWidget(process_btn)
        
        # Export button
        self.export_demand_btn = QPushButton("Export Results")
        self.export_demand_btn.clicked.connect(self.export_demand_forecast)
        self.export_demand_btn.setMinimumHeight(40)
        self.export_demand_btn.setEnabled(False)
        self.export_demand_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #808080;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }
            QPushButton:enabled {
                background-color: #34495e;
                color: #e0e0e0;
            }
            QPushButton:enabled:hover {
                background-color: #415364;
            }
        """)
        layout.addWidget(self.export_demand_btn)
        
        # Chart
        self.chart = MatplotlibWidget()
        layout.addWidget(self.chart)
        
        # Metrics
        metrics_layout = QHBoxLayout()
        self.avg_metric = QLabel("Average Daily Demand: -")
        self.total_metric = QLabel("Total Demand: -")
        self.peak_metric = QLabel("Peak Demand: -")
        
        for i, metric in enumerate([self.avg_metric, self.total_metric, self.peak_metric]):
            colors = ['#2a2416', '#232328', '#2e1a1e']
            borders = ['#f39c12', '#95a5a6', '#e74c3c']
            metric.setStyleSheet(f"""
                QLabel {{
                    background-color: {colors[i]};
                    border-left: 4px solid {borders[i]};
                    padding: 20px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 500;
                    color: #e0e0e0;
                }}
            """)
            metrics_layout.addWidget(metric)
        
        layout.addLayout(metrics_layout)
        
        # Data Summary Stats Panel
        stats_panel = QFrame()
        stats_panel.setStyleSheet("""
            QFrame {
                background-color: #2a2416;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        stats_layout = QVBoxLayout()
        
        stats_title = QLabel("Analysis Summary")
        stats_title.setStyleSheet("QLabel { color: #f39c12; font-size: 14px; font-weight: bold; }")
        stats_layout.addWidget(stats_title)
        
        self.stats_info = QLabel("Select a product and date to view demand statistics")
        self.stats_info.setStyleSheet("QLabel { color: #c0c0c0; font-size: 12px; }")
        self.stats_info.setWordWrap(True)
        stats_layout.addWidget(self.stats_info)
        
        stats_panel.setLayout(stats_layout)
        layout.addWidget(stats_panel)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def process_demand(self):
        try:
            item = self.product_combo.currentText()
            # Convert QDate to Python date object
            qdate = self.date_edit.date()
            start_date = date(qdate.year(), qdate.month(), qdate.day())
            
            # Forecast now applies date-specific adjustments including seasonality,
            # day of week effects, and weekly patterns based on the start_date.
            
            item_forecasted_demand, predictions_df = forecast_demand(start_date, 30, item)
            
            if predictions_df.empty or predictions_df['predicted_mean'].isna().all():
                QMessageBox.warning(self, "Warning", "Forecast data is empty or contains only NaNs.")
                return
            
            # Plot
            self.chart.clear()
            ax = self.chart.figure.add_subplot(111, facecolor='#2a2a2a')
            
            # Plot line with markers for better visibility
            ax.plot(predictions_df.index, predictions_df['predicted_mean'], 
                   color='#9b59b6', linewidth=2.5, marker='o', markersize=4, 
                   markerfacecolor='#a29bfe', markeredgecolor='#9b59b6', 
                   markeredgewidth=1.5, alpha=0.9)
            
            # Fill area under the curve for better visualization
            ax.fill_between(predictions_df.index, predictions_df['predicted_mean'], 
                           alpha=0.2, color='#9b59b6')
            
            # Format x-axis with dates
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(predictions_df)//10)))
            
            # Rotate and align the tick labels for better readability
            self.chart.figure.autofmt_xdate(rotation=45, ha='right')
            
            ax.set_xlabel('Date', color='#e0e0e0', fontsize=11, fontweight='bold')
            ax.set_ylabel('Demand (Units)', color='#e0e0e0', fontsize=11, fontweight='bold')
            ax.set_title(f'Demand Forecast for {item}', color='#ffffff', fontsize=14, fontweight='bold', pad=15)
            ax.tick_params(colors='#808080', labelsize=9)
            ax.grid(True, alpha=0.3, color='#404040', linestyle='--', linewidth=0.8)
            ax.spines['bottom'].set_color('#404040')
            ax.spines['top'].set_color('#404040')
            ax.spines['left'].set_color('#404040')
            ax.spines['right'].set_color('#404040')
            
            # Add tight layout to prevent label cutoff
            self.chart.figure.tight_layout()
            self.chart.canvas.draw()
            
            # Update metrics
            avg_demand = round(predictions_df['predicted_mean'].mean(), 0)
            total_demand = int(item_forecasted_demand)
            peak_demand = round(predictions_df['predicted_mean'].max(), 2)
            
            self.avg_metric.setText(f"Average Daily Demand: {avg_demand}")
            self.total_metric.setText(f"Total Demand: {total_demand}")
            self.peak_metric.setText(f"Peak Demand: {peak_demand}")
            
            # Store forecast for export
            self.demand_data = predictions_df
            self.current_item = item
            self.export_demand_btn.setEnabled(True)
            
            # Update stats panel
            self.stats_info.setText(
                f"Product: {item}\n"
                f"Forecast Period: {start_date.strftime('%Y-%m-%d')} to {(start_date + pd.Timedelta(days=29)).strftime('%Y-%m-%d')}\n"
                f"Days Forecasted: 30\n"
                f"Note: Forecast includes seasonal, weekly, and day-of-week patterns\n"
                f"specific to the selected start date."
            )
            
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Model file not found. Please train models first.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def export_demand_forecast(self):
        if not hasattr(self, 'demand_data'):
            return
        
        try:
            # Export CSV
            file_path, _ = QFileDialog.getSaveFileName(self, "Export Demand Forecast", f"demand_forecast_{self.current_item}.csv", "CSV Files (*.csv)")
            if file_path:
                self.demand_data.to_csv(file_path)
                
                # Also save chart
                chart_path = file_path.replace('.csv', '.png')
                self.chart.figure.savefig(chart_path, dpi=300, facecolor='#1a1a1a', edgecolor='none')
                
                QMessageBox.information(self, "Success", f"Exported to:\n{file_path}\n{chart_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting: {str(e)}")


class VisualizationPage(QWidget):
    """Data Visualization Page"""
    def __init__(self):
        super().__init__()
        self.df = None
        self.df_ungrouped = None
        self.init_ui()
        self.load_data()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Data Visualization Dashboard")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Analysis type selection
        analysis_layout = QHBoxLayout()
        analysis_layout.addWidget(QLabel("Analysis Type:"))
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(["General Graphs", "Specified Analysis"])
        self.analysis_combo.currentTextChanged.connect(self.on_analysis_changed)
        analysis_layout.addWidget(self.analysis_combo)
        analysis_layout.addStretch()
        layout.addLayout(analysis_layout)
        
        # Stacked widget for different analysis types
        self.stacked_widget = QStackedWidget()
        
        # General Graphs Page
        general_page = QWidget()
        general_layout = QVBoxLayout()
        
        plot_selection = QHBoxLayout()
        plot_selection.addWidget(QLabel("Plot Type:"))
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(['Select plot type', 'Bar Plot', 'Line Plot', 'Pie Chart', 
                                   'Box Plot', 'Histogram', 'Scatter Plot'])
        self.plot_combo.currentTextChanged.connect(self.on_plot_changed)
        plot_selection.addWidget(self.plot_combo)
        plot_selection.addStretch()
        general_layout.addLayout(plot_selection)
        
        col_selection = QHBoxLayout()
        col_selection.addWidget(QLabel("Column 1:"))
        self.col1_combo = QComboBox()
        col_selection.addWidget(self.col1_combo)
        
        col_selection.addWidget(QLabel("Column 2:"))
        self.col2_combo = QComboBox()
        self.col2_combo.setVisible(False)
        col_selection.addWidget(self.col2_combo)
        col_selection.addStretch()
        general_layout.addLayout(col_selection)
        
        # Time range filter (always visible)
        time_filter = QHBoxLayout()
        time_filter.addWidget(QLabel("Filter by Date:"))
        
        from PyQt6.QtCore import QDate
        self.time_start = QDateEdit()
        self.time_start.setCalendarPopup(True)
        self.time_start.setDisplayFormat("yyyy-MM-dd")
        self.time_start.setDate(QDate.currentDate().addYears(-1))  # Default to 1 year ago
        time_filter.addWidget(self.time_start)
        
        time_filter.addWidget(QLabel("to"))
        
        self.time_end = QDateEdit()
        self.time_end.setCalendarPopup(True)
        self.time_end.setDisplayFormat("yyyy-MM-dd")
        self.time_end.setDate(QDate.currentDate())  # Default to today
        time_filter.addWidget(self.time_end)
        
        self.enable_time_filter = QCheckBox("Enable Date Filter")
        self.enable_time_filter.setChecked(False)
        time_filter.addWidget(self.enable_time_filter)
        
        time_filter.addStretch()
        general_layout.addLayout(time_filter)
        
        gen_buttons = QHBoxLayout()
        generate_btn = QPushButton("Generate Plot")
        generate_btn.clicked.connect(self.generate_general_plot)
        generate_btn.setMinimumHeight(40)
        generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        
        self.export_general_btn = QPushButton("Export Chart")
        self.export_general_btn.clicked.connect(lambda: self.export_chart(self.general_chart))
        self.export_general_btn.setMinimumHeight(40)
        self.export_general_btn.setEnabled(False)
        self.export_general_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #808080;
                padding: 10px 20px;
                font-size: 14px;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }
            QPushButton:enabled {
                background-color: #34495e;
                color: #e0e0e0;
            }
            QPushButton:enabled:hover {
                background-color: #415364;
            }
        """)
        
        gen_buttons.addWidget(generate_btn)
        gen_buttons.addWidget(self.export_general_btn)
        gen_buttons.addStretch()
        general_layout.addLayout(gen_buttons)
        
        self.general_chart = MatplotlibWidget()
        general_layout.addWidget(self.general_chart)
        general_page.setLayout(general_layout)
        
        # Specified Analysis Page
        specified_page = QWidget()
        specified_layout = QVBoxLayout()
        
        specified_selection = QHBoxLayout()
        specified_selection.addWidget(QLabel("Analytics:"))
        self.specified_combo = QComboBox()
        self.specified_combo.addItems([
            'Select Analytics',
            'Top 20 Best-selling products',
            'Order Size & Value Analysis',
            'Seasonal Revenue Patterns by Year',
            'seasonal Patterns'
        ])
        specified_selection.addWidget(self.specified_combo)
        specified_selection.addStretch()
        specified_layout.addLayout(specified_selection)
        
        spec_buttons = QHBoxLayout()
        specified_btn = QPushButton("Generate Analysis")
        specified_btn.clicked.connect(self.generate_specified_analysis)
        specified_btn.setMinimumHeight(40)
        specified_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        
        self.export_specified_btn = QPushButton("Export Chart")
        self.export_specified_btn.clicked.connect(lambda: self.export_chart(self.specified_chart))
        self.export_specified_btn.setMinimumHeight(40)
        self.export_specified_btn.setEnabled(False)
        self.export_specified_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #808080;
                padding: 10px 20px;
                font-size: 14px;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }
            QPushButton:enabled {
                background-color: #34495e;
                color: #e0e0e0;
            }
            QPushButton:enabled:hover {
                background-color: #415364;
            }
        """)
        
        spec_buttons.addWidget(specified_btn)
        spec_buttons.addWidget(self.export_specified_btn)
        spec_buttons.addStretch()
        specified_layout.addLayout(spec_buttons)
        
        self.specified_chart = MatplotlibWidget()
        specified_layout.addWidget(self.specified_chart)
        specified_page.setLayout(specified_layout)
        
        self.stacked_widget.addWidget(general_page)
        self.stacked_widget.addWidget(specified_page)
        layout.addWidget(self.stacked_widget)
        
        # System Status Dashboard
        status_panel = QFrame()
        status_panel.setStyleSheet("""
            QFrame {
                background-color: #1a2332;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        status_layout = QHBoxLayout()
        
        # Data Status
        data_status = QLabel("Data Loaded")
        data_status.setStyleSheet("""
            QLabel {
                color: #27ae60;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                background-color: #1a2e26;
                border-radius: 5px;
            }
        """)
        
        # Orders Info
        self.records_info = QLabel(f"Orders: {len(self.df):,}" if self.df is not None else "No Data")
        self.records_info.setStyleSheet("""
            QLabel {
                color: #4a90e2;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                background-color: #1a2332;
                border-radius: 5px;
            }
        """)
        
        # Charts Ready
        charts_ready = QLabel("Visualizations Ready")
        charts_ready.setStyleSheet("""
            QLabel {
                color: #9b59b6;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                background-color: #232328;
                border-radius: 5px;
            }
        """)
        
        status_layout.addWidget(data_status)
        status_layout.addWidget(self.records_info)
        status_layout.addWidget(charts_ready)
        status_layout.addStretch()
        
        status_panel.setLayout(status_layout)
        layout.addWidget(status_panel)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def load_data(self):
        try:
            self.df = pd.read_csv(resource_path("data/depi_grouped.csv"), low_memory=False)
            self.df_ungrouped = pd.read_csv(resource_path("data/depi_ungrouped.csv"), low_memory=False)
            
            # Debug: Check data loaded correctly
            print(f"Loaded grouped CSV: {len(self.df)} rows")
            if 'item_count' in self.df.columns:
                print(f"Item count range in loaded data: {self.df['item_count'].min()} to {self.df['item_count'].max()}")
            
            self.df1, self.df2 = split_numeric_categorical(self.df)
            self.df1_ungrouped, self.df2_ungrouped = split_numeric_categorical(self.df_ungrouped)
            
            # Populate column combos
            self.col1_combo.addItems(self.df.columns.tolist())
            self.col2_combo.addItems(self.df.columns.tolist())
            
            # Convert timestamp column to datetime for both dataframes
            if 'created_at' in self.df.columns:
                print(f"Converting created_at column. Sample values: {self.df['created_at'].head()}")
                self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
                self.df_ungrouped['created_at'] = pd.to_datetime(self.df_ungrouped['created_at'], errors='coerce')
                
                # Remove any NaT (Not a Time) values before getting min/max
                valid_dates = self.df['created_at'].dropna()
                
                if len(valid_dates) > 0:
                    # Set date range to first and last date in the data
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    
                    print(f"Min date: {min_date}, Max date: {max_date}")
                    
                    from PyQt6.QtCore import QDate
                    # Set the date pickers to the full data range
                    q_min_date = QDate(min_date.year, min_date.month, min_date.day)
                    q_max_date = QDate(max_date.year, max_date.month, max_date.day)
                    
                    self.time_start.setDate(q_min_date)
                    self.time_end.setDate(q_max_date)
                    
                    # Also set the min/max limits so users can't select outside the data range
                    self.time_start.setMinimumDate(q_min_date)
                    self.time_start.setMaximumDate(q_max_date)
                    self.time_end.setMinimumDate(q_min_date)
                    self.time_end.setMaximumDate(q_max_date)
                    
                    print(f"Date pickers set: {q_min_date.toString('yyyy-MM-dd')} to {q_max_date.toString('yyyy-MM-dd')}")
                else:
                    print("No valid dates found in created_at column")
            
            # Update orders info label
            self.records_info.setText(f"Orders: {len(self.df):,}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
    
    def on_analysis_changed(self, text):
        if text == "General Graphs":
            self.stacked_widget.setCurrentIndex(0)
        else:
            self.stacked_widget.setCurrentIndex(1)
    
    def on_plot_changed(self, plot_type):
        if plot_type in ['Bar Plot', 'Line Plot', 'Scatter Plot']:
            self.col2_combo.setVisible(True)
        else:
            self.col2_combo.setVisible(False)
        
        # Use numeric columns for Box Plot and Histogram
        if plot_type in ['Box Plot', 'Histogram']:
            self.col1_combo.clear()
            self.col1_combo.addItems(self.df1.columns.tolist())
        else:
            self.col1_combo.clear()
            self.col1_combo.addItems(self.df.columns.tolist())
    
    def generate_general_plot(self):
        plot_type = self.plot_combo.currentText()
        if plot_type == 'Select plot type':
            return
        
        col1 = self.col1_combo.currentText()
        col2 = self.col2_combo.currentText() if self.col2_combo.isVisible() else None
        
        self.general_chart.clear()
        
        try:
            # Filter data by time range if checkbox is enabled
            df_filtered = self.df.copy()
            df_ungrouped_filtered = self.df_ungrouped.copy()
            
            if self.enable_time_filter.isChecked() and 'created_at' in df_filtered.columns:
                from PyQt6.QtCore import QDate
                start_date = pd.Timestamp(self.time_start.date().toPyDate())
                end_date = pd.Timestamp(self.time_end.date().toPyDate())
                
                print(f"Filtering data from {start_date} to {end_date}")
                print(f"Before filter - grouped: {len(df_filtered)}, ungrouped: {len(df_ungrouped_filtered)}")
                
                # Apply time filter to both dataframes
                df_filtered = df_filtered[(df_filtered['created_at'] >= start_date) & 
                                         (df_filtered['created_at'] <= end_date)]
                df_ungrouped_filtered = df_ungrouped_filtered[(df_ungrouped_filtered['created_at'] >= start_date) & 
                                                              (df_ungrouped_filtered['created_at'] <= end_date)]
                
                print(f"After filter - grouped: {len(df_filtered)}, ungrouped: {len(df_ungrouped_filtered)}")
            
            # Create matplotlib plots directly instead of using Streamlit functions
            ax = self.general_chart.figure.add_subplot(111)
            
            if plot_type == 'Bar Plot':
                grouped = df_ungrouped_filtered.groupby(col1)[col2].sum()
                x_pos = range(len(grouped))
                ax.bar(x_pos, grouped.values, color='#4a90e2', alpha=0.8, edgecolor='#2a2a2a')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(grouped.index, rotation=90, ha='center', fontsize=8)
                ax.set_xlabel(col1, fontsize=10, color='#e0e0e0', labelpad=10)
                ax.set_ylabel(col2, fontsize=10, color='#e0e0e0')
                ax.set_title(f'{col2} by {col1}', fontsize=12, color='#ffffff')
                # Add extra bottom margin for rotated labels
                self.general_chart.figure.subplots_adjust(bottom=0.25)
            elif plot_type == 'Line Plot':
                ax.plot(df_filtered[col1], df_filtered[col2], marker='o', linestyle='-')
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f'{col2} vs {col1}')
            elif plot_type == 'Pie Chart':
                data = df_filtered[col1].value_counts()
                ax.pie(data.values, labels=data.index, autopct='%1.1f%%')
                ax.set_title(f'Distribution of {col1}')
            elif plot_type == 'Box Plot':
                self.df.boxplot(column=col1, ax=ax)
                ax.set_ylabel(col1)
                ax.set_title(f'Box Plot of {col1}')
            elif plot_type == 'Histogram':
                # Ensure the column is numeric and drop NaN values
                data = pd.to_numeric(df_filtered[col1], errors='coerce').dropna()
                if len(data) > 0:
                    ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
                    ax.set_xlabel(col1)
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Histogram of {col1}')
                else:
                    ax.text(0.5, 0.5, 'No numeric data available', 
                           ha='center', va='center', transform=ax.transAxes)
            elif plot_type == 'Scatter Plot':
                ax.scatter(df_filtered[col1], df_filtered[col2], alpha=0.6)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f'{col2} vs {col1}')
            
            style_dark_axis(ax)
            self.general_chart.figure.tight_layout(pad=2.0)
            self.general_chart.canvas.draw()
            self.export_general_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating plot: {str(e)}")
    
    def generate_specified_analysis(self):
        analysis = self.specified_combo.currentText()
        if analysis == 'Select Analytics':
            return
        
        self.specified_chart.clear()
        
        try:
            ax = self.specified_chart.figure.add_subplot(111)
            
            if analysis == 'Top 20 Best-selling products':
                # Create top 20 plot without Streamlit
                df_copy = self.df_ungrouped.copy()
                df_copy['Lineitem name'] = df_copy['Lineitem name'].apply(
                    lambda x: x.split(',') if isinstance(x, str) else []
                )
                exploded = df_copy.explode('Lineitem name')
                exploded['Lineitem quantity'] = pd.to_numeric(exploded['Lineitem quantity'], errors='coerce')
                exploded = exploded.dropna(subset=['Lineitem quantity'])
                top_products = exploded.groupby('Lineitem name')['Lineitem quantity'].sum().sort_values(ascending=False).head(20)
                
                ax.barh(range(len(top_products)), top_products.values, color='royalblue')
                ax.set_yticks(range(len(top_products)))
                ax.set_yticklabels(top_products.index)
                ax.set_xlabel('Number of Orders')
                ax.set_ylabel('Product Name')
                ax.set_title('Top 20 Best-Selling Products')
                ax.invert_yaxis()
                
            elif analysis == 'Order Size & Value Analysis':
                # Order analysis matching the original streamlit version
                if 'item_count' in self.df.columns and 'Total' in self.df.columns:
                    print(f"Before processing: {len(self.df)} rows, item_count range: {self.df['item_count'].min()} to {self.df['item_count'].max()}")
                    
                    df_temp = self.df[['item_count', 'Total']].copy()
                    print(f"After selecting columns: {len(df_temp)} rows")
                    
                    df_temp = df_temp.dropna(subset=['item_count', 'Total'])
                    print(f"After dropna: {len(df_temp)} rows")
                    
                    df_temp['item_count'] = pd.to_numeric(df_temp['item_count'], errors='coerce')
                    df_temp['Total'] = pd.to_numeric(df_temp['Total'], errors='coerce')
                    print(f"After to_numeric: {len(df_temp)} rows, item_count range: {df_temp['item_count'].min()} to {df_temp['item_count'].max()}")
                    
                    df_temp = df_temp.dropna()
                    print(f"After final dropna: {len(df_temp)} rows, item_count range: {df_temp['item_count'].min()} to {df_temp['item_count'].max()}")
                    
                    if len(df_temp) > 0:
                        # Create 2 subplots
                        self.specified_chart.figure.clear()
                        ax1 = self.specified_chart.figure.add_subplot(211)
                        ax2 = self.specified_chart.figure.add_subplot(212)
                        
                        # Plot 1: Order size distribution
                        order_size_counts = df_temp['item_count'].value_counts().sort_index()
                        ax1.bar(order_size_counts.index, order_size_counts.values, color='royalblue', alpha=0.7)
                        ax1.set_xlabel('Number of Items in Order')
                        ax1.set_ylabel('Number of Orders')
                        ax1.set_title('Distribution of Items per Order')
                        style_dark_axis(ax1)
                        
                        # Plot 2: Average order value by item count
                        avg_order_value = df_temp.groupby('item_count')['Total'].mean()
                        ax2.plot(avg_order_value.index, avg_order_value.values, 
                                marker='o', linestyle='-', color='#f39c12', linewidth=2, markersize=8)
                        ax2.set_xlabel('Number of Items in Order')
                        ax2.set_ylabel('Average Order Value ($)')
                        ax2.set_title('Average Order Value by Number of Items')
                        style_dark_axis(ax2)
                        
                        # Don't call ax.grid() here since we have ax1 and ax2
                        self.specified_chart.figure.tight_layout()
                        self.specified_chart.canvas.draw()
                        return  # Exit early since we handled the figure differently
                    else:
                        ax.text(0.5, 0.5, 'No valid order data available', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=12)
                else:
                    ax.text(0.5, 0.5, 'Required columns not found:\nitem_count, Total', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                
            elif analysis == 'Seasonal Revenue Patterns by Year':
                # Seasonal revenue by year - matching original with year, season, Total columns
                if 'year' in self.df.columns and 'season' in self.df.columns and 'Total' in self.df.columns:
                    df_temp = self.df.dropna(subset=['year', 'season', 'Total']).copy()
                    df_temp['Total'] = pd.to_numeric(df_temp['Total'], errors='coerce')
                    df_temp = df_temp.dropna(subset=['Total'])
                    
                    if len(df_temp) > 0:
                        # Group by year and season
                        seasonal_orders = df_temp.groupby(['year', 'season'])['Total'].sum().unstack(fill_value=0)
                        
                        # Plot grouped bar chart
                        x = np.arange(len(seasonal_orders.index))
                        width = 0.2
                        seasons = seasonal_orders.columns.tolist()
                        
                        for i, season in enumerate(seasons):
                            offset = width * (i - len(seasons)/2 + 0.5)
                            ax.bar(x + offset, seasonal_orders[season], width, label=season, alpha=0.8)
                        
                        ax.set_xlabel('Year')
                        ax.set_ylabel('Total Revenue')
                        ax.set_title('Seasonal Revenue Patterns by Year')
                        ax.set_xticks(x)
                        ax.set_xticklabels(seasonal_orders.index)
                        ax.legend(title='Season')
                        ax.grid(True, alpha=0.3, axis='y')
                    else:
                        ax.text(0.5, 0.5, 'No seasonal data available', 
                               ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, 'Required columns not found:\nyear, season, Total', 
                           ha='center', va='center', transform=ax.transAxes)
                
            elif analysis == 'seasonal Patterns':
                # Comprehensive seasonal analysis - matching original with multiple subplots
                if all(col in self.df.columns for col in ['year', 'season', 'Name', 'Total', 'year_month']):
                    df_temp = self.df.dropna(subset=['Total', 'Name']).copy()
                    df_temp['Total'] = pd.to_numeric(df_temp['Total'], errors='coerce')
                    df_temp = df_temp.dropna(subset=['Total'])
                    
                    if len(df_temp) > 0:
                        # Create 4 subplots
                        self.specified_chart.figure.clear()
                        ax1 = self.specified_chart.figure.add_subplot(411)
                        ax2 = self.specified_chart.figure.add_subplot(412)
                        ax3 = self.specified_chart.figure.add_subplot(413)
                        ax4 = self.specified_chart.figure.add_subplot(414)
                        
                        # 1. Total revenue by year
                        yearly_revenue = df_temp.groupby('year')['Total'].sum()
                        ax1.bar(yearly_revenue.index, yearly_revenue.values, color='#27ae60', alpha=0.8)
                        ax1.set_ylabel('Revenue')
                        ax1.set_title('Total Revenue by Year')
                        style_dark_axis(ax1)
                        
                        # 2. Average order value by season
                        seasonal_avg = df_temp.groupby('season')['Total'].mean()
                        ax2.plot(seasonal_avg.index, seasonal_avg.values, marker='o', color='#f39c12', linewidth=2, markersize=8)
                        ax2.set_ylabel('Avg Order Value')
                        ax2.set_title('Average Order Value by Season')
                        style_dark_axis(ax2)
                        
                        # 3. Total revenue by season
                        seasonal_total = df_temp.groupby('season')['Total'].sum()
                        ax3.plot(seasonal_total.index, seasonal_total.values, marker='o', color='#e74c3c', linewidth=2, markersize=8)
                        ax3.set_ylabel('Total Revenue')
                        ax3.set_title('Total Revenue by Season')
                        style_dark_axis(ax3)
                        
                        # 4. Monthly revenue trend
                        monthly_trend = df_temp.groupby('year_month')['Total'].sum()
                        ax4.plot(range(len(monthly_trend)), monthly_trend.values, marker='o', color='#4a90e2', linewidth=1.5)
                        ax4.set_xlabel('Year-Month')
                        ax4.set_ylabel('Total Revenue')
                        ax4.set_title('Monthly Revenue Trend')
                        ax4.set_xticks(range(0, len(monthly_trend), max(1, len(monthly_trend)//10)))
                        ax4.set_xticklabels(monthly_trend.index[::max(1, len(monthly_trend)//10)], rotation=45, ha='right')
                        style_dark_axis(ax4)
                        
                        self.specified_chart.figure.tight_layout()
                        self.specified_chart.canvas.draw()
                        return  # Exit early
                    else:
                        ax.text(0.5, 0.5, 'No seasonal data available', 
                               ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, 'Required columns not found:\nyear, season, Name, Total, year_month', 
                           ha='center', va='center', transform=ax.transAxes)
            
            style_dark_axis(ax)
            self.specified_chart.figure.tight_layout()
            self.specified_chart.canvas.draw()
            self.export_specified_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating analysis: {str(e)}")
    
    def export_chart(self, chart_widget):
        try:
            file_path, _ = QFileDialog.getSaveFileName(self, "Export Chart", "chart.png", "PNG Files (*.png);;PDF Files (*.pdf)")
            if file_path:
                chart_widget.figure.savefig(file_path, dpi=300, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Chart exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting chart: {str(e)}")


class RetrainingPage(QWidget):
    """Model Retraining Page"""
    def __init__(self):
        super().__init__()
        self.new_data = None
        self.new_marketing = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Re-Training Models")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Upload new data
        data_section = QHBoxLayout()
        data_lbl = QLabel("New Data:")
        data_lbl.setStyleSheet("QLabel { color: #e0e0e0; font-size: 14px; }")
        data_section.addWidget(data_lbl)
        self.data_label = QLabel("No file selected")
        self.data_label.setStyleSheet("""
            QLabel {
                color: #808080;
                padding: 8px;
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
        """)
        data_btn = QPushButton("Upload CSV")
        data_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #e0e0e0;
                padding: 8px 16px;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #333333;
                border: 1px solid #4a90e2;
            }
        """)
        data_btn.clicked.connect(lambda: self.upload_file("data"))
        data_section.addWidget(data_btn)
        data_section.addWidget(self.data_label)
        data_section.addStretch()
        layout.addLayout(data_section)
        
        # Data preview
        self.data_table = QTableWidget()
        self.data_table.setMaximumHeight(150)
        data_preview_lbl = QLabel("Data Preview:")
        data_preview_lbl.setStyleSheet("QLabel { color: #e0e0e0; font-size: 13px; padding-top: 10px; }")
        layout.addWidget(data_preview_lbl)
        layout.addWidget(self.data_table)
        
        # Upload marketing data
        marketing_section = QHBoxLayout()
        marketing_lbl = QLabel("Marketing Data:")
        marketing_lbl.setStyleSheet("QLabel { color: #e0e0e0; font-size: 14px; }")
        marketing_section.addWidget(marketing_lbl)
        self.marketing_label = QLabel("No file selected")
        self.marketing_label.setStyleSheet("""
            QLabel {
                color: #808080;
                padding: 8px;
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
        """)
        marketing_btn = QPushButton("Upload CSV")
        marketing_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #e0e0e0;
                padding: 8px 16px;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #333333;
                border: 1px solid #4a90e2;
            }
        """)
        marketing_btn.clicked.connect(lambda: self.upload_file("marketing"))
        marketing_section.addWidget(marketing_btn)
        marketing_section.addWidget(self.marketing_label)
        marketing_section.addStretch()
        layout.addLayout(marketing_section)
        
        # Marketing preview
        self.marketing_table = QTableWidget()
        self.marketing_table.setMaximumHeight(150)
        marketing_preview_lbl = QLabel("Marketing Data Preview:")
        marketing_preview_lbl.setStyleSheet("QLabel { color: #e0e0e0; font-size: 13px; padding-top: 10px; }")
        layout.addWidget(marketing_preview_lbl)
        layout.addWidget(self.marketing_table)
        
        # Train button
        train_btn = QPushButton("Start Training")
        train_btn.clicked.connect(self.train_models)
        train_btn.setMinimumHeight(50)
        train_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #c0392b, stop:1 #a93226);
            }
        """)
        layout.addWidget(train_btn)
        
        # Progress log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        log_lbl = QLabel("Training Log:")
        log_lbl.setStyleSheet("QLabel { color: #e0e0e0; font-size: 13px; padding-top: 10px; }")
        layout.addWidget(log_lbl)
        layout.addWidget(self.log)
        
        # Model Training History Panel
        history_panel = QFrame()
        history_panel.setStyleSheet("""
            QFrame {
                background-color: #2a1a1a;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        history_layout = QVBoxLayout()
        
        history_title = QLabel("Training History & Guidelines")
        history_title.setStyleSheet("QLabel { color: #e74c3c; font-size: 14px; font-weight: bold; }")
        history_layout.addWidget(history_title)
        
        # Check for existing model versions
        try:
            directory = resource_path("data/")
            pattern = r"depi_v(\d+)\.csv"
            version_numbers = []
            for filename in os.listdir(directory):
                match = re.match(pattern, filename)
                if match:
                    version_numbers.append(int(match.group(1)))
            
            if version_numbers:
                highest_version = max(version_numbers)
                history_text = (
                    f"Current Data Version: v{highest_version}\n"
                    f"Available Versions: {len(version_numbers)}\n\n"
                    "Training Guidelines:\n"
                    "• Ensure data formats match existing structure\n"
                    "• Training may take 5-15 minutes depending on data size\n"
                    "• Models will be saved automatically upon completion\n"
                    "• New version will be incremented to v" + str(highest_version + 1)
                )
            else:
                history_text = "No previous training versions found.\nThis will create the first model version."
        except:
            history_text = "Unable to check training history. Ensure data directory exists."
        
        self.history_info = QLabel(history_text)
        self.history_info.setStyleSheet("QLabel { color: #c0c0c0; font-size: 12px; line-height: 1.6; }")
        self.history_info.setWordWrap(True)
        history_layout.addWidget(self.history_info)
        
        history_panel.setLayout(history_layout)
        layout.addWidget(history_panel)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def upload_file(self, file_type):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                df = pd.read_csv(file_path)
                
                if file_type == "data":
                    self.new_data = df
                    self.data_label.setText(f"Loaded: {os.path.basename(file_path)}")
                    self.display_dataframe(self.data_table, df.head())
                else:
                    self.new_marketing = df
                    self.marketing_label.setText(f"Loaded: {os.path.basename(file_path)}")
                    self.display_dataframe(self.marketing_table, df.head())
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    
    def display_dataframe(self, table, df):
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.tolist())
        
        for i in range(len(df)):
            for j in range(len(df.columns)):
                value = df.iloc[i, j]
                table.setItem(i, j, QTableWidgetItem(str(value)))
    
    def train_models(self):
        if self.new_data is None or self.new_marketing is None:
            QMessageBox.warning(self, "Warning", "Please upload both data files first!")
            return
        
        try:
            self.log.append("Processing data...")
            
            # Find highest version
            directory = resource_path("data/")
            pattern = r"depi_v(\d+)\.csv"
            version_numbers = []
            
            for filename in os.listdir(directory):
                match = re.match(pattern, filename)
                if match:
                    version_numbers.append(int(match.group(1)))
            
            highest_version = max(version_numbers) if version_numbers else 0
            highest_version_file = f"depi_v{highest_version}.csv"
            
            # Load and combine data
            df_old = pd.read_csv(os.path.join(directory, highest_version_file))
            df_combined = pd.concat([self.new_data, df_old], ignore_index=True)
            
            # Save new version
            new_version = highest_version + 1
            new_version_file = f"depi_v{new_version}.csv"
            df_combined.to_csv(os.path.join(directory, new_version_file), index=False)
            self.log.append(f"Saved combined data as {new_version_file}")
            
            # Process data
            self.log.append("Preprocessing data...")
            df_demand = preprocess_ungrouped(df_combined)
            df_series = grouping_data(df_demand)
            df_series = feature_engineering(df_series)
            df_series = time_series_features(df_series)
            df_demand = handle_category(df_demand)
            
            # Combine marketing data
            df_old_marketing = pd.read_csv(resource_path("data/marketing_data.csv"))
            df_marketing_combined = pd.concat([df_old_marketing, self.new_marketing], ignore_index=True)
            
            # Start training in background thread
            self.log.append("Starting model training...")
            self.training_thread = TrainingThread(df_demand, df_series, df_marketing_combined)
            self.training_thread.progress.connect(self.log.append)
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing data: {str(e)}")
            self.log.append(f"ERROR: {str(e)}")
    
    def on_training_finished(self, success, message):
        self.log.append(message)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)


class MainWindow(QMainWindow):
    """Main Application Window"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Sales Forecasting & On-Demand Predictions")
        self.setGeometry(100, 100, 1600, 950)
        
        # Set application-wide dark theme style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #e0e0e0;
            }
            QPushButton {
                border-radius: 5px;
                font-weight: 500;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QLabel {
                color: #e0e0e0;
            }
            QComboBox {
                padding: 8px;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                background: #2a2a2a;
                color: #e0e0e0;
                min-height: 25px;
            }
            QComboBox:hover {
                border: 1px solid #4a90e2;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #e0e0e0;
                selection-background-color: #4a90e2;
                border: 1px solid #3a3a3a;
            }
            QDateEdit {
                padding: 8px;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                background: #2a2a2a;
                color: #e0e0e0;
                min-height: 25px;
            }
            QDateEdit::drop-down {
                border: none;
            }
            QCalendarWidget {
                background-color: #2a2a2a;
                color: #e0e0e0;
            }
            QTableWidget {
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                background-color: #2a2a2a;
                color: #e0e0e0;
                gridline-color: #3a3a3a;
            }
            QTableWidget::item {
                padding: 5px;
                color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #333333;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #4a4a4a;
                font-weight: bold;
                color: #e0e0e0;
            }
            QTextEdit {
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                background-color: #2a2a2a;
                color: #e0e0e0;
                padding: 8px;
            }
            QMessageBox {
                background-color: #2a2a2a;
                color: #e0e0e0;
            }
            QMessageBox QLabel {
                color: #e0e0e0;
                padding: 10px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Sidebar
        sidebar = QFrame()
        sidebar.setFrameShape(QFrame.Shape.NoFrame)
        sidebar.setMaximumWidth(280)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #0d0d0d;
                border: none;
                border-right: 1px solid #2a2a2a;
            }
        """)
        
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(5)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)
        
        # Logo/Title
        logo_container = QWidget()
        logo_layout = QVBoxLayout()
        logo_layout.setSpacing(5)
        
        app_title = QLabel("Sales & Demand\nForecasting")
        app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 22px;
                font-weight: bold;
                padding: 20px 10px 10px 10px;
                letter-spacing: 1px;
            }
        """)
        
        subtitle = QLabel("Analytics Platform")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 12px;
                padding-bottom: 20px;
                text-transform: uppercase;
                letter-spacing: 2px;
            }
        """)
        
        logo_layout.addWidget(app_title)
        logo_layout.addWidget(subtitle)
        logo_container.setLayout(logo_layout)
        sidebar_layout.addWidget(logo_container)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("QFrame { background-color: #2a2a2a; max-height: 1px; }")
        sidebar_layout.addWidget(line)
        
        sidebar_layout.addSpacing(10)
        
        # Navigation buttons
        self.nav_buttons = []
        nav_items = [
            "Sales Forecasting",
            "Demand Analysis",
            "Data Visualizations",
            "Model Training"
        ]
        
        for i, text in enumerate(nav_items):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setMinimumHeight(50)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #c0c0c0;
                    text-align: left;
                    padding: 15px 20px;
                    border: none;
                    font-size: 15px;
                    border-radius: 0px;
                    margin: 0px;
                }
                QPushButton:hover {
                    background-color: #1a1a1a;
                    color: #ffffff;
                }
                QPushButton:checked {
                    background-color: #1a1a1a;
                    border-left: 3px solid #4a90e2;
                    color: #ffffff;
                    font-weight: 600;
                }
            """)
            btn.clicked.connect(lambda checked, idx=i: self.change_page(idx))
            self.nav_buttons.append(btn)
            sidebar_layout.addWidget(btn)
        
        sidebar_layout.addStretch()
        
        sidebar.setLayout(sidebar_layout)
        
        # Content area with margin
        content_container = QWidget()
        content_container.setStyleSheet("QWidget { background-color: #1a1a1a; }")
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(25, 25, 25, 25)
        
        self.content_stack = QStackedWidget()
        self.content_stack.addWidget(SalesForecastingPage())
        self.content_stack.addWidget(DemandProductPage())
        self.content_stack.addWidget(VisualizationPage())
        self.content_stack.addWidget(RetrainingPage())
        
        content_layout.addWidget(self.content_stack)
        content_container.setLayout(content_layout)
        
        # Add to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_container, 1)
        
        central_widget.setLayout(main_layout)
        
        # Set initial page
        self.nav_buttons[0].setChecked(True)
        self.change_page(0)
    
    def change_page(self, index):
        self.content_stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application-wide palette for buttons
    palette = app.palette()
    palette.setColor(palette.ColorRole.Button, QColor('#4a90e2'))
    palette.setColor(palette.ColorRole.ButtonText, QColor('#ffffff'))
    palette.setColor(palette.ColorRole.Highlight, QColor('#4a90e2'))
    palette.setColor(palette.ColorRole.HighlightedText, QColor('#ffffff'))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Required for PyInstaller on macOS/Windows
    main()
