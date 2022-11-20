<h1>Streamlit Application ReadMe</h1>

<b>The following libraries must be installed in order for the application to run:</b>
<ul>
    <li>pandas</li>
    <li>numpy</li>
    <li>pyarrow</li>
    <li>matplotlib</li>
    <li>datetime</li>
    <li>json</li>
    <li>gzip</li>
    <li>os</li>
    <li>codecs</li>
    <li>csv</li>
    <li>io</li>
    <li>re</li>
    <li>string</li>
    <li>glob</li>
    <li>pynvml</li>
    <li>tqdm</li>
    <li>seaborn</li>
    <li>xlsxwriter</li>
    <li>nltk</li>
    <li>spacy</li>
    <li>streamlit</li>
    <li>streamlit_autorefresh</li>
    <li>huggingface</li>
    <li>tensorflow</li>
    <li>torch</li>
    <li>email</li>
</ul>

<b>Add Penn State One Drive folder OR re-create the directory</b>
<ol>
    <li>Penn State One Drive</li>
    <ol>
        <li>First, download the One Drive desktop app</li>
        <li>Open the app and sign in to your Penn State Account</li>
        <li>Navigate to "WC DAAN 888 FA22 - Team-8/Documents"</li>
        <li>Right-click the "Team-8" folder and select "Add shortcut to OneDrive"</li>
    </ol>
    <li>Re-create directory</li>
    <ol>
        <li>In your user drive (C:\Users\&lt;username&gt;), create a folder called "Live Demo"</li>
        <li>In "Live Demo" create a folder called "PowerBI" and a folder called "Streamlitfiles"</li>
    </ol>
</ol>

<b>To run the application locally:</b>
<ol>
    <li>Open Conda Terminal</li>
    <li>Switch to the environment that all of the libraries are installed on</li>
    <li>Navigate to the location of the WorkableModelStreamlit.py file</li>
    <li>Run the following command: streamlit run "python file location"</li>
    <li>A new tab should pop up on your preferred browser</li>
    <li><span style="color:red">IMPORTANT: When the little running man is in the top right corner, allow him to finish before proceeding</span></li>
    <li><span style="color:red">One other note, while the application is running or about to run, do not open Live Demo/clasified_file.csv or Live Demo/PowerBI/classifiedfile.xlsx</span></li>
    <li>To execute the application, there are 3 separate paths:</li>
    <ol>
        <li><b>Free-text entry</b></li>
        <li>The only required fields are Overall Star Rating and the free-text</li>
        <li>Once entered, select the "Check the data!" button</li>
        <li>Once the running man has disappeared, the final record output can be found in the "Live Demo/Streamlitfiles" folder which is also appended to the classified_file files (csv and xlsx)</li>
    </ol>
    <ol>
        <li><b>Bulk file load</b></li>
        <li>The file must include a column called "reviewText" in order to run</li>
        <li>A file template is available for download at the bottom of the application and includes some sample data</li>
        <li>Do not fill out any of the free-text fields, just load the file (must be in csv format)</li>
        <li>Once entered, select the "Check the data!" button</li>
        <li>Once the running man has disappeared, the final record(s) output can be found in the "Live Demo/Streamlitfiles" folder which is also appended to the classified_file files (csv and xlsx)</li>
    </ol>
    <ol>
        <li><b>Free-text and bulk file load</b></li>
        <li>Follow the same steps as above, filling in both the free-text and uploading a bulk file</li>
    </ol>
</ol>

<b>Check application output emails</b>
<ol>
    <li>Reach out to ajb6058@psu.edu for login details</li>
</ol>

<b>PowerBI</b>
<ol>
    <li>First download the PowerBI file and open PowerBI Desktop</li>
    <li>Open the PowerBI file and navigate to "Data" on the left-hand nav</li>
    <li>Find the data-source on the right-hand nav called "classified_file", right-click and edit query</li>
    <li>If you are utilizing the OneDrive route:</li>
    <ol>
        <li>In the power query editor pop-up on the left-hand nav, right-click "classified_file" and select "advanced editor"</li>
        <li>Paste the following:</li>
        <ol>
            <li>let<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;Source = SharePoint.Files("https://pennstateoffice365.sharepoint.com/sites/DAAN888FA22-Team-8/", [ApiVersion = 15]),<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;#"Filtered Rows" = Table.SelectRows(Source, each ([Name] = "combined_data.xlsx")),<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;#"combined_data xlsx_https://pennstateoffice365 sharepoint com/sites/DAAN888FA22-Team-8/Shared Documents/Team-8/Live Demo/PowerBI/" = #"Filtered Rows"{[Name="combined_data.xlsx",#"Folder Path"="https://pennstateoffice365.sharepoint.com/sites/DAAN888FA22-Team-8/Shared Documents/Team-8/Live Demo/PowerBI/"]}[Content],<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;#"Imported Excel Workbook" = Excel.Workbook(#"combined_data xlsx_https://pennstateoffice365 sharepoint com/sites/DAAN888FA22-Team-8/Shared Documents/Team-8/Live Demo/PowerBI/"),<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;Sheet1_Sheet = #"Imported Excel Workbook"{[Item="Sheet1",Kind="Sheet"]}[Data],<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;#"Promoted Headers" = Table.PromoteHeaders(Sheet1_Sheet, [PromoteAllScalars=true]),<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;#"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{{"Column1", Int64.Type}, {"idx", Int64.Type}, {"OriginalIndex", Int64.Type}, {"title", type text}, {"brand", type text}, {"main_cat", type text}, {"price", type text}, {"asin", type text}, {"verified", type logical}, {"reviewTime", type date}, {"reviewText", type text}, {"summary", type text}, {"overall", Int64.Type}, {"sub_category", type text}, {"price_adj", type number}, {"original_text", type text}, {"original_summary", type text}, {"sentiment", type text}, {"Class 0", type text}, {"Rating 0", type text}, {"Class 1", type text}, {"Rating 1", type text}, {"Class 2", type text}, {"Rating 2", type text}, {"Class 3", type text}, {"Rating 3", type text}, {"Class 4", type text}, {"Rating 4", type text}, {"Classification", type text}, {"Shipping", Int64.Type}, {"PricingFinance", Int64.Type}, {"CustomerService", Int64.Type}, {"ProductQuality", Int64.Type}, {"Validation", Int64.Type}, {"BERT_FullSentiment", type text}, {"BERT_FullScore", type number}, {"originalEntry", type text}, {"Classification1", type text}, {"Classification2", type text}, {"Classification3", type text}, {"Classification4", type text}, {"Classification5", type text}, {"emailList", type text}})<br />
                in<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;#"Changed Type"</li>
        </ol>
        <li>Click "done" then in the top left, close and apply</li>
    </ol>
    <li>If you are utilizing the local directory route:</li>
    <ol>
        <li>There will be an error on the screen, select the "Go to Error" button and "edit settings"</li>
        <li>Replace the file path with your local directory (C:\Users\&lt;username&gt;\Live Demo\PowerBI\combined_data.xlsx)</li>
        <li>In the top left, close and apply</li>
    </ol>
</ol>