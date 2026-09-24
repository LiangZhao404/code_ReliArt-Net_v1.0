function T=loadMetadata(csvFile)
% Required columns: image_path, catalogue_label, fidelity_target, fidelity_mask, artwork_family, source
T=readtable(csvFile,TextType="string");
req=["image_path","catalogue_label","fidelity_target","fidelity_mask"];
assert(all(ismember(req,string(T.Properties.VariableNames))),"Metadata CSV is missing required columns.");
end
