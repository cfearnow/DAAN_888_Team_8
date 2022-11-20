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
        <li>open the app and sign in to your Penn State Account</li>
        <li>navigate to "WC DAAN 888 FA22 - Team-8/Documents"</li>
        <li>right-click the "Team-8" folder and select "Add shortcut to OneDrive"</li>
    </ol>
    <li>Re-create directory</li>
    <ol>
        <li>in your user drive (C:\Users\<username>), create a folder called "Live Demo"</li>
        <li>in "Live Demo" create a folder called "PowerBI" and a folder called "Streamlitfiles"</li>
    </ol>
</ol>

<b>To run the application locally:</b>
<ol>
    <li>Open Conda Terminal</li>
    <li>Switch to the environment that all of the libraries are installed on</li>
    <li>Navigate to the location of the WorkableModelStreamlit.py file</li>
    <li>Run the following command: streamlit run "python file location"</li>
    <li>A new tab should pop up on your preferred browser</li>
    <li>IMPORTANT: When the little running man is in the top right corner, allow him to finish before proceeding</li>
    <li>One other note, while the application is running or about to run, do not open Live Demo/clasified_file.csv or Live Demo/PowerBI/classifiedfile.xlsx</li>
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
        <li>paste the following:</li>
        <ol>
            <li>let
                    Source = SharePoint.Files("https://pennstateoffice365.sharepoint.com/sites/DAAN888FA22-Team-8/", [ApiVersion = 15]),
                    #"Filtered Rows" = Table.SelectRows(Source, each ([Name] = "combined_data.xlsx")),
                    #"combined_data xlsx_https://pennstateoffice365 sharepoint com/sites/DAAN888FA22-Team-8/Shared Documents/Team-8/Live Demo/PowerBI/" = #"Filtered Rows"{[Name="combined_data.xlsx",#"Folder Path"="https://pennstateoffice365.sharepoint.com/sites/DAAN888FA22-Team-8/Shared Documents/Team-8/Live Demo/PowerBI/"]}[Content],
                    #"Imported Excel Workbook" = Excel.Workbook(#"combined_data xlsx_https://pennstateoffice365 sharepoint com/sites/DAAN888FA22-Team-8/Shared Documents/Team-8/Live Demo/PowerBI/"),
                    Sheet1_Sheet = #"Imported Excel Workbook"{[Item="Sheet1",Kind="Sheet"]}[Data],
                    #"Promoted Headers" = Table.PromoteHeaders(Sheet1_Sheet, [PromoteAllScalars=true]),
                    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{{"Column1", Int64.Type}, {"idx", Int64.Type}, {"OriginalIndex", Int64.Type}, {"title", type text}, {"brand", type text}, {"main_cat", type text}, {"price", type text}, {"asin", type text}, {"verified", type logical}, {"reviewTime", type date}, {"reviewText", type text}, {"summary", type text}, {"overall", Int64.Type}, {"sub_category", type text}, {"price_adj", type number}, {"original_text", type text}, {"original_summary", type text}, {"sentiment", type text}, {"Class 0", type text}, {"Rating 0", type text}, {"Class 1", type text}, {"Rating 1", type text}, {"Class 2", type text}, {"Rating 2", type text}, {"Class 3", type text}, {"Rating 3", type text}, {"Class 4", type text}, {"Rating 4", type text}, {"Classification", type text}, {"Shipping", Int64.Type}, {"PricingFinance", Int64.Type}, {"CustomerService", Int64.Type}, {"ProductQuality", Int64.Type}, {"Validation", Int64.Type}, {"BERT_FullSentiment", type text}, {"BERT_FullScore", type number}, {"originalEntry", type text}, {"Classification1", type text}, {"Classification2", type text}, {"Classification3", type text}, {"Classification4", type text}, {"Classification5", type text}, {"emailList", type text}})
                in
                    #"Changed Type"</li>
        </ol>
        <li>Click "done" then in the top left, close and apply</li>
    </ol>
    <li>If you are utilizing the local directory route:</li>
    <ol>
        <li>There will be an error on the screen, select the "Go to Error" button and "edit settings"</li>
        <li>replace the file path with your local directory (C:\Users\<username>\Live Demo\PowerBI\combined_data.xlsx)</li>
        <li>in the top left, close and apply</li>
    </ol>
</ol>