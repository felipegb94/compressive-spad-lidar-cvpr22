function [json_dict] = load_json(filepath)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    fid = fopen(filepath); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    json_dict = jsondecode(str);
end

