  # soil-erosion-classifier
soil-erosion-class-finder/
├── app.R                # Shiny app code
├── README.md            # Project description
├── data/                # Data files (e.g., rasters, shapefiles)
│   ├── NDVI.tif
│   ├── SAVI.tif
│   ├── BI.tif
│   ├── SBI.tif
│   ├── Slope.tif
│   ├── LS_Factor.tif
│   ├── TWI.tif
│   └── soil_data.shp    # Sample shapefile for ground truth
├── outputs/             # Output files (e.g., predictions, models)
│   ├── classification_result_tuned.tif
│   ├── confusion_matrix.csv
│   └── random_forest_tuned_model.rds
├── .gitignore           # Files to exclude from GitHub
└── scripts/           
library(raster)
library(leaflet)
library(shiny)
library(caret)
library(sf)
library(sp)
library(dplyr)
library(pROC)
library(terra)
library(PROJ)

# Set working directory
working_directory <- choose.dir()
setwd(working_directory)

# Load the multi-band satellite image
satellite_data <- brick(file.choose())

# Access bands by name or index
red_band <- satellite_data[["B4"]]   # Red band
nir_band <- satellite_data[["B8"]]   # NIR band
green_band <- satellite_data[["B3"]] # Green band
blue_band <- satellite_data[["B2"]]  # Blue band

# NDVI calculation
ndvi <- (nir_band - red_band) / (nir_band + red_band)
writeRaster(ndvi, "NDVI.tif", overwrite = TRUE)

# SAVI calculation
L <- 0.5
savi <- ((nir_band - red_band) / (nir_band + red_band + L)) * (1 + L)
writeRaster(savi, "SAVI.tif", overwrite = TRUE)

# BI calculation
bi <- sqrt((red_band^2 + green_band^2) / 2)
writeRaster(bi, "BI.tif", overwrite = TRUE)

# SBI calculation
sbi <- sqrt(red_band^2 + green_band^2 + blue_band^2)
writeRaster(sbi, "SBI.tif", overwrite = TRUE)

# Load DEM
dem <- raster(file.choose())

# Generate terrain parameters
slope <- terrain(dem, opt = "slope", unit = "degrees")
writeRaster(slope, "Slope.tif", overwrite = TRUE)

# Flow Accumulation (simplified example)
flow_accumulation <- calc(dem, fun = function(x) {
  ifelse(x >= 1, 1, 0)
})
writeRaster(flow_accumulation, "flow_accumulation.tif", overwrite = TRUE)

# LS Factor calculation
slope_radians <- terrain(dem, opt = "slope", unit = "radians")
ls_factor <- (flow_accumulation + 1)^0.4 * sin(slope_radians)^1.3
writeRaster(ls_factor, "LS_Factor.tif", overwrite = TRUE)

# TWI calculation
twi <- log((flow_accumulation + 1) / tan(slope_radians))
writeRaster(twi, "TWI.tif", overwrite = TRUE) and library(raster)
# Smooth indices using focal operations
focal_ndvi_mean <- focal(ndvi, w = matrix(1, nrow = 3, ncol = 3), fun = mean, na.rm = TRUE)
focal_savi_mean <- focal(savi, w = matrix(1, nrow = 3, ncol = 3), fun = mean, na.rm = TRUE)
focal_bi_mean <- focal(bi, w = matrix(1, nrow = 3, ncol = 3), fun = mean, na.rm = TRUE)
focal_sbi_mean <- focal(sbi, w = matrix(1, nrow = 3, ncol = 3), fun = mean, na.rm = TRUE)

# Align all rasters
reference_raster <- ndvi
savi <- resample(savi, reference_raster, method = "bilinear")
bi <- resample(bi, reference_raster, method = "bilinear")
sbi <- resample(sbi, reference_raster, method = "bilinear")
slope <- resample(slope, reference_raster, method = "bilinear")
ls_factor <- resample(ls_factor, reference_raster, method = "bilinear")
twi <- resample(twi, reference_raster, method = "bilinear")

# Crop rasters to the same extent
extent_to_use <- extent(reference_raster)
focal_ndvi_mean <- crop(focal_ndvi_mean, extent_to_use)
focal_savi_mean <- crop(focal_savi_mean, extent_to_use)
focal_bi_mean <- crop(focal_bi_mean, extent_to_use)
focal_sbi_mean <- crop(focal_sbi_mean, extent_to_use)
slope <- crop(slope, extent_to_use)
ls_factor <- crop(ls_factor, extent_to_use)
twi <- crop(twi, extent_to_use)

# Combine into raster stack
input_stack <- stack(focal_ndvi_mean, focal_savi_mean, focal_bi_mean, focal_sbi_mean, slope, ls_factor, twi)

# Load ROI shapefile and ground truth data
roi <- st_read(file.choose())
soilData_file <- file.choose()
gt_data <- read.csv(soilData_file)

# Convert ground truth data to spatial object
gt_data <- st_as_sf(gt_data, coords = c(2,1), crs = 4326)
gt_data <- st_transform(gt_data, crs(input_stack))

# Extract raster values at ground truth points
train_data <- extract(input_stack, gt_data)
train_data <- data.frame(train_data)
train_data$class <- as.factor(gt_data$class)
# Remove rows with missing values
train_data <- na.omit(train_data)
if (nrow(train_data) == 0) {
  stop("All rows contain missing values after na.omit(). Check your raster layers and ground truth data.")
}
# Train-test split
set.seed(123)
trainIndex <- createDataPartition(train_data$class, p = 0.8, list = FALSE)
trainSet <- train_data[trainIndex, ]
testSet <- train_data[-trainIndex, ]


# Hyperparameter tuning and training Random Forest
tunegrid <- expand.grid(mtry = c(2, 3, 4, 5))
rf_tuned_model <- train(
  class ~ .,
  data = trainSet,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = tunegrid,
  metric = "Accuracy"
)

# Evaluate model performance
rf_pred <- predict(rf_tuned_model, newdata = testSet)
conf_matrix <- confusionMatrix(rf_pred, testSet$class)
kappa_val <- conf_matrix$overall['Kappa']
accuracy_val <- conf_matrix$overall['Accuracy']
f1_val <- mean(conf_matrix$byClass[, "F1"])

# Save metrics and model
write.csv(conf_matrix$table, "confusion_matrix.csv", row.names = TRUE)
saveRDS(rf_tuned_model, "random_forest_tuned_model.rds")

# Generate predicted classification map
predicted_map <- predict(input_stack, rf_tuned_model)
writeRaster(predicted_map, "classification_result_tuned.tif", overwrite = TRUE)
# Downsample raster to reduce size for Shiny app
predicted_map_downsampled <- aggregate(predicted_map, fact = 4, fun = mean)
writeRaster(predicted_map_downsampled, "Classification_Map_Downsampled.tif", overwrite = TRUE)

# Load downsampled raster for Shiny app
predicted_map <- raster("Classification_Map_Downsampled.tif")

# Shiny App
ui <- fluidPage(
  titlePanel("Soil Erosion Class Finder"),
  sidebarLayout(
    sidebarPanel(
      h4("Erosion Class:"),
      verbatimTextOutput("erosion_class"),
      h4("Model Performance:"),
      verbatimTextOutput("model_metrics")
    ),
    mainPanel(
      leafletOutput("map")
    )
  )
)

server <- function(input, output, session) {
  erosion_class <- reactiveVal("Click on the map to see the erosion class.")
  
  output$erosion_class <- renderText({
    erosion_class()
  })
  
  output$model_metrics <- renderText({
    paste("Accuracy:", round(accuracy_val, 3), "\nKappa:", round(kappa_val, 3))
  })
  
  output$map <- renderLeaflet({
    leaflet() %>%
      addTiles() %>%
      addRasterImage(predicted_map, colors = terrain.colors(10), opacity = 0.7)
  })
  
  observeEvent(input$map_click, {
    click <- input$map_click
    lat <- click$lat
    lon <- click$lng
    point <- SpatialPoints(cbind(lon, lat), proj4string = CRS(projection(predicted_map)))
    value <- extract(predicted_map, point)
    erosion_class(ifelse(is.na(value), "No data available.", paste("Erosion Class:", value)))
  })
}

shinyApp(ui, server) 
git clone https://github.com/<your-username>/soil-erosion-class-finder.git
