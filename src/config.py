import os

# Path to the folder containing the "Archive" folder
_BASE_DIRECTORY = r"C:\Users\jtorres\OneDrive\consumos_uclm\Datos"

# Path of the excel containing CUPS and locations
EXCEL_CUPS_LOCATION_PATH = os.path.join(_BASE_DIRECTORY, "CUPS_UCLM_TODOS.xlsx")

# Path to the root folder containing the data
_ROOT_PATH_DATA = os.path.join(_BASE_DIRECTORY, "Archive")
# Path to folder containing all folders with hourly data
CURVAS_HORARIAS_PATH = os.path.join(_ROOT_PATH_DATA, "Curvas horarias")
# Path to folder containing all folders with hourly data which have been reindexed
CURVAS_HORARIAS_TRATADAS_PATH = os.path.join(_ROOT_PATH_DATA, "Curvas horarias tratadas")
# Path to folder containing all folders with quarter hourly data
CURVAS_CUARTO_HORARIAS_PATH = os.path.join(_ROOT_PATH_DATA, "Curvas cuarto horarias")
# Path to folder containing all folders with quarter hourly data which have been reindexed
CURVAS_CUARTO_HORARIAS_TRATADAS_PATH = os.path.join(_ROOT_PATH_DATA, "Curvas cuarto horarias tratadas")
# Path to folder containing all folders with invoices
INVOICES_PATH = os.path.join(_ROOT_PATH_DATA, "Facturas")

NEXUS_NAME = "Nexus"
IBERDROLA_NAME = "Iberdrola"

LOGIN_MARKETER = "webpage"
USER_MARKETER = "user"
PASS_MARKETER = "pass"
