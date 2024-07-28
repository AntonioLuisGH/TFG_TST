function mat_to_csv(mat_file, csv_file)
    % Load the data from the .mat file
    load(mat_file, 'val');
    
    % Eliminate the fields we are not interested in
    fields_to_eliminate = {'rid', 'cid','lid','dd1','wtlux','alux','ecb1','ecp1','st2', 'p2', 'ec2', 'vwc2', 'ecb2', 'ecp2', 'st3', 'p3', 'ec3', 'vwc3', 'ecb3', 'ecp3','dli'};
    val_cleaned = val; % Create a copy of the original structure
    for i = 1:numel(fields_to_eliminate)
        val_cleaned = rmfield(val_cleaned, fields_to_eliminate{i});
    end
    
    % Convert the data from a structure to a table
    val_table = struct2table(val_cleaned);
    
    % Change the field names of the table
    new_names = {'Identificator', 'Date', 'Month','Day','Year','Hour','Minute','Second','Temperature','Relative_humidity','Vapor_Pressure_Deficit', 'Light', 'Soil_temperature', 'Permittivity', 'Electroconductivity', 'Volumetric_water_content', 'Diameter', 'Photosynthetically_Active_Radiation','Battery_voltage'};
    val_table.Properties.VariableNames = new_names;
    
    % Write the data to a .csv file
    writetable(val_table, csv_file, 'Delimiter', ';');
end

