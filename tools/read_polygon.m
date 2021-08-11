% Read a .shp file to write files for each polygon described
S = shaperead('data/zones/scallop_composite.shp');
nPolygons = length(S);

for iPolygon = 1:nPolygons
    boundary = [S(iPolygon).X; S(iPolygon).Y].';
    writematrix(boundary(1:end-1, :), sprintf('data/zones/boundary%0.f.txt', iPolygon), 'delimiter', ' ');
end

