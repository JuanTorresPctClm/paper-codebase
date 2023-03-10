from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
import tabula


def compose_folder_name(year, month):
    """
    Composes the name of the folder containing the data for a given
    year and month
    :param year:
    :param month:
    :return: str in the form yyyymm
    """
    year = str(year)
    month = str(month).zfill(2)

    return f"{year}{month}"


def filter_by_cups(df, cups):
    """
    Filters a dataframe to present only the given cups
    :param df: dataframe to filter
    :param cups: cups to be left on df
    :return: pd.Dataframe
    """
    return df[df['CUPS'].str.startswith(cups)].copy()


def convert_pdf_to_string(file_path):
    """
    Parses a pdf to string
    Original function: https://towardsdatascience.com/pdf-text-extraction-in-python-5b6ab9e92dd
    :param file_path: absolute path to pdf file
    :return:
    """
    output_string = StringIO()
    with open(file_path, 'rb') as opened_file:
        parser = PDFParser(opened_file)
        pdf_doc = PDFDocument(parser)
        resource_manager = PDFResourceManager()
        text_converter = TextConverter(resource_manager, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(resource_manager, text_converter)
        for page in PDFPage.create_pages(pdf_doc):
            interpreter.process_page(page)

    return output_string.getvalue()


def extract_tables_from_pdf(file_path, guess=True):
    return tabula.read_pdf(file_path, pages='all', guess=guess)


def get_first_index_of_column_of_dataframe_starting_with(df, column_name, text):
    """
    Searches in the specified column of the pandas DataFrame the first index that starts with text.
    NOTE: THE INDEX OF THE DATAFRAME MUST BE CORRELATIVE AND START IN 0
    """
    # Check which rows start with the given text
    mask = list(df[column_name].str.startswith(text))

    # loop over the list to find the first appearance
    counter = 0
    for i in mask:
        if i is True:
            return counter
        counter += 1

    return None
