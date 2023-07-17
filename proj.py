from flask import Flask, render_template, request, jsonify
from datetime import datetime
import numpy as np
import pandas as pd
import io
import base64
import psycopg2
from datacube import Datacube
import odc.algo
from deafrica_tools.bandindices import calculate_indices
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('proj.html')

@app.route('/result', methods=['POST'])
def result():

    image_base64=[]
    i_base64=' '
    hist_base64=[]
    h_base64=' '
    c=["Red","Green","Blue","Magenta","Cyan","Yellow","Purple","Orange","Teal","Pink","Coral","SaddleBrown","LawnGreen","DarkOrchid", "DarkOrange"]

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    date_start = request.form['date-start']
    date_end = request.form['date-end']
    method = request.form['metric']
    # lat_start = request.form['lat-start']
    # lon_start = request.form['lon-start']
    # lat_end = request.form['lat-end']
    # lon_end = request.form['lon-end']


    lat_starts = request.form.get('lat-start').split(';')[:-1]
    lon_starts = request.form.get('lon-start').split(';')[:-1]
    lat_ends = request.form.get('lat-end').split(';')[:-1]
    lon_ends = request.form.get('lon-end').split(';')[:-1]

    # Convert the lists of strings to floats
    lat_starts = [float(lat) for lat in lat_starts]
    lon_starts = [float(lon) for lon in lon_starts]
    lat_ends = [float(lat) for lat in lat_ends]
    lon_ends = [float(lon) for lon in lon_ends]

    print("------------------------------------------------------------------",lat_starts)
    print("------------------------------------------------------------------",lon_starts)
    print("------------------------------------------------------------------",lat_ends)
    print("------------------------------------------------------------------",lon_ends)
    print(len(lat_starts))

    j=0
    k=0

    # Iterate over the areas
    for i in range(len(lat_starts)):


        lat_start = lat_starts[i]
        lon_start = lon_starts[i]
        lat_end = lat_ends[i]
        lon_end = lon_ends[i]

        dc = Datacube(app="04_Plotting")

        lat_range = (lat_start, lat_end)
        lon_range = (lon_start, lon_end)
        time_range = (date_start, date_end)


        ds = dc.load(
            product="s2a_sen2cor_granule",
            measurements=["B04_10m", "B03_10m", "B08_10m"],
            x=lon_range,
            y=lat_range,
            time=time_range,
            output_crs='EPSG:6933',
            resolution=(-30, 30)
        )

        dataset = odc.algo.to_f32(ds)
        print(ds)

        if not ds:
            return jsonify({"message":"null"})
        
        
        # Process each area and plot the corresponding graph
        # ...


        print(method)

    
        if method == 'ndvi':


            print("------------------------------------ndvi method executing------------------------------------------")
            band_diff = dataset.B08_10m - dataset.B04_10m
            band_sum = dataset.B08_10m + dataset.B04_10m
            ndvi = band_diff / band_sum


            print(ndvi)


            print("============================================================================")
                   
            
            if(len(ds['time'])==1):
                plt.figure()
                ndvi.plot( cmap="YlGn",vmin=-1, vmax=1 ,figsize=(8, 4))

                # Convert the plot to a PNG image in memory
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                plt.close()
                
                # buffer.close()

                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                # Encode the PNG image as a base64 string
                i_base64 = base64.b64encode(buffer.read()).decode('utf-8')


                image_base64.append(i_base64)
                
            elif(len(ds['time'])>1):
                # Generate some data for plotting

                plt.figure()

                # Create a figure and subplots with specified width ratios
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4), gridspec_kw={'width_ratios': [1, 1, 0.1]})

                # Plot on the first subplot
                plot1 = ndvi[0].plot(ax=axes[0], cmap="YlGn",vmin=-1, vmax=1, col_wrap=3)

                # Plot on the second subplot
                plot2 = ndvi[-1].plot(ax=axes[1],cmap="YlGn", vmin=-1, vmax=1, col_wrap=3)


                axes[0].set_title(str(ndvi.time.values[0]).split('T')[0])
                axes[1].set_title(str(ndvi.time.values[-1]).split('T')[0])

                # Create a custom legend patch
                legend_patch = Patch(color=c[i], label='Selection-'+str(i+1))

                # Add the legend patch to the third subplot
                axes[2].legend(handles=[legend_patch], loc='center')

                # Hide the axis of the third subplot
                axes[2].axis('off')

                # Adjust spacing between subplots
                plt.subplots_adjust(wspace=0.3)
                # Display the figure
                # plt.show()

                # Convert the plot to a PNG image in memory
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                plt.close()
                
                # buffer.close()

                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                # Encode the PNG image as a base64 string
                i_base64 = base64.b64encode(buffer.read()).decode('utf-8')


                image_base64.append(i_base64)




            print("-----------------------------------------------------------------")


            ndvi.plot.hist(bins=1000, range=(-1,1), facecolor=c[i], figsize=(9, 4))
            plt.xlabel("NDVI")
            plt.ylabel("Number of Pixels")
            plt.title('NDVI Histogram of'+' Selection-'+str(i+1)+' of color-'+c[i])



            # Convert the plot to a PNG image in memory
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            plt.close()

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # Encode the PNG image as a base64 string
            h_base64 = base64.b64encode(buffer.read()).decode('utf-8')


            hist_base64.append(h_base64)




        elif method == 'ndwi':

        
            print("------------------------------------ndwi method executing------------------------------------------")
            band_diff = dataset.B08_10m - dataset.B03_10m
            band_sum = dataset.B08_10m + dataset.B03_10m
            ndwi = band_diff / band_sum
            print(ndwi)

            plt.figure()

            if(len(ds['time'])==1):
                ndwi.plot( cmap="Blues",vmin=-1, vmax=1 ,figsize=(8, 4))
            elif(len(ds['time'])>1):
                # Generate some data for plotting


                plt.figure()


                # Create a figure and subplots with specified width ratios
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4), gridspec_kw={'width_ratios': [1, 1, 0.1]})

                # Plot on the first subplot
                plot1 = ndwi[0].plot(ax=axes[0], cmap="Blues", vmin=-1, vmax=1, col_wrap=3)

                # Plot on the second subplot
                plot2 = ndwi[-1].plot(ax=axes[1], cmap="Blues",vmin=-1, vmax=1, col_wrap=3)


                axes[0].set_title(str(ndwi.time.values[0]).split('T')[0])
                axes[1].set_title(str(ndwi.time.values[-1]).split('T')[0])

                # Create a custom legend patch
                legend_patch = Patch(color=c[i], label='Selection-'+str(i+1))

                # Add the legend patch to the third subplot
                axes[2].legend(handles=[legend_patch], loc='center')

                # Hide the axis of the third subplot
                axes[2].axis('off')

                # Adjust spacing between subplots
                plt.subplots_adjust(wspace=0.3)
                # Display the figure
                # plt.show()

            # Convert the plot to a PNG image in memory
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            plt.close()

            # buffer.close()

            # Encode the PNG image as a base64 string
            i_base64 = base64.b64encode(buffer.read()).decode('utf-8')


            image_base64.append(i_base64)



            print("-----------------------------------------------------------------")
            ndwi.plot.hist(bins=1000, range=(-1,1), facecolor=c[i], figsize=(9, 4))
            plt.xlabel("NDWI")
            plt.ylabel("Number of Pixels")
            plt.title('NDWI Histogram of'+' Selection-'+str(i+1)+' of color-'+c[i])
            # Convert the plot to a PNG image in memory
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            plt.close()

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # Encode the PNG image as a base64 string
            h_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            hist_base64.append(h_base64)




        elif method == 'forest':

            print("------------------------------------forest method executing------------------------------------------")
            band_diff = dataset.B08_10m - dataset.B04_10m
            band_sum = dataset.B08_10m + dataset.B04_10m
            ndvi = band_diff / band_sum
            

            # Extract the time values from the xarray object
            time_values = ndvi.time.values

            # Add the time values as a new column in the dataframe

            # Get the spatial resolution of the dataset
            spatial_resolution = np.abs(ds.geobox.affine.a)

            # Calculate the area per pixel
            area_per_pixel = spatial_resolution**2

            # Determine the number of pixels in the dataset
            num_pixels = ds.sizes['x'] * ds.sizes['y']

            # Calculate the total area
            total_area = area_per_pixel * num_pixels


            total_area_km2=total_area/1000000

            # Print the results
            print("Area per pixel: {} square meters".format(area_per_pixel))
            print("Total area: {} square meters".format(total_area))
            print("Total area: {} square kms".format(total_area_km2))



            dense_forest_mask = np.where((ndvi > 0.6) & (ndvi < 0.8), 1, 0)
            open_forest_mask = np.where((ndvi > 0.3) & (ndvi < 0.6) , 1, 0)
            x=np.where((ndvi>0.8) | (ndvi<0.1),1,0)

            sparse_forest_mask = np.where((ndvi > 0.1) & (ndvi < 0.3) , 1, 0)
            
            w=np.sum(x[0])
            print(np.sum(x[0]),"---",area_per_pixel)
            ta=area_per_pixel*w
            ta2=ta/1000000
            print(ta,ta2)
            # print(dense_forest_mask,' sum = ',np.sum(dense_forest_mask[0]),' len = ',len(dense_forest_mask[0]))
            # print(open_forest_mask,' sum = ',np.sum(open_forest_mask[1]),' len = ',len(open_forest_mask[1]))
            # print(sparse_forest_mask,' sum = ',np.sum(sparse_forest_mask[1]),' len = ',len(sparse_forest_mask[1]))

            def area( a ):
                print(np.sum(a))
                xw=area_per_pixel*np.sum(a)
                print(xw)
                return xw/1000000
            d=['time','dfm','ofm','sfm','tfa']
            x=[]
            for i in range(len(dense_forest_mask)):
                w=[]
                w.append(pd.to_datetime(time_values[i]))
                w.append(area(dense_forest_mask[i]))
                w.append(area(open_forest_mask[i]))
                w.append(area(sparse_forest_mask[i]))
                w.append(area(dense_forest_mask[i])+area(open_forest_mask[i])+area(sparse_forest_mask[i]))
                x.append(w)
            # df['time'] = pd.to_datetime(time_values)
            df = pd.DataFrame(x, columns=d)
            df


            # Assuming your time column is named 'time' and the value column is named 'ndvi'
            # Convert the 'time' column to pandas timetime if it's not already in that format
            # Read the CSV file into a pandas DataFrame

            df['time'] = pd.to_datetime(df['time'])
            print(df.head())
            X_train=df['time']
            y_train=df['tfa']
            # Split the data into training and test sets
            # X_train, X_test, y_train, y_test = train_test_split(df['time'], df['dfm'],test_size=0.3,shuffle=False)

            # Extract the time components as features
            X_train_features = pd.DataFrame()
            X_train_features['year'] = X_train.dt.year
            X_train_features['month'] = X_train.dt.month
            X_train_features['day'] = X_train.dt.day
            # Add more features as per your requirements



        

            # Initialize and fit the Random Forest Regressor model
            model = RandomForestRegressor()
            model.fit(X_train_features, y_train)

            # Extract features from the test data
            X_test_features = pd.DataFrame()
            X_test_features['year'] = [2025]
            X_test_features['month'] =[ 5]
            X_test_features['day'] = [5]
            # Add more features as per your requirements
            print(X_train_features.head())
            print(X_test_features.head())
            # Predict the values
            predictions = model.predict(X_test_features)

            # Print the predictions
            print(predictions)


            prede=model.predict(X_train_features)
            print(prede)


            indices = np.arange(len(X_train))

            df['date'] = pd.to_datetime(df['time'])

            # Plot the actual values
            plt.plot(df['date'], df['tfa'], color=c[j], label='Actual')

            # Plot the predicted values
            # plt.plot(df['date'], prede, color='red', label='Predicted')

            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Forest Area(sq_km)')
            plt.title('Forest Area Cover - Actual '+' Selection-'+str(j+1)+' of color-'+c[j])
            j=j+1

            # Add legend
            plt.legend()

            # Display the graph
            # # plt.show()



            # Convert the plot to a PNG image in memory
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0) 

            plt.close()

            # Encode the PNG image as a base64 string
            i_base64 = base64.b64encode(buffer.read()).decode('utf-8')


            image_base64.append(i_base64)


        elif method == 'ml':


            print("------------------------------------ml method executing------------------------------------------")
            band_diff = dataset.B08_10m - dataset.B04_10m
            band_sum = dataset.B08_10m + dataset.B04_10m
            ndvi = band_diff / band_sum
            

            # Extract the time values from the xarray object
            time_values = ndvi.time.values

            # Add the time values as a new column in the dataframe


            # Get the spatial resolution of the dataset
            spatial_resolution = np.abs(ds.geobox.affine.a)

            # Calculate the area per pixel
            area_per_pixel = spatial_resolution**2

            # Determine the number of pixels in the dataset
            num_pixels = ds.sizes['x'] * ds.sizes['y']

            # Calculate the total area
            total_area = area_per_pixel * num_pixels


            total_area_km2=total_area/1000000

            # Print the results
            print("Area per pixel: {} square meters".format(area_per_pixel))
            print("Total area: {} square meters".format(total_area))
            print("Total area: {} square kms".format(total_area_km2))


            dense_forest_mask = np.where((ndvi > 0.6) & (ndvi < 0.8), 1, 0)
            open_forest_mask = np.where((ndvi > 0.3) & (ndvi < 0.6) , 1, 0)
            x=np.where((ndvi>0.8) | (ndvi<0.1),1,0)

            sparse_forest_mask = np.where((ndvi > 0.1) & (ndvi < 0.3) , 1, 0)
            w=np.sum(x[0])
            print(np.sum(x[0]),"---",area_per_pixel)
            ta=area_per_pixel*w
            ta2=ta/1000000
            print(ta,ta2)
            # print(dense_forest_mask,' sum = ',np.sum(dense_forest_mask[0]),' len = ',len(dense_forest_mask[0]))
            # print(open_forest_mask,' sum = ',np.sum(open_forest_mask[1]),' len = ',len(open_forest_mask[1]))
            # print(sparse_forest_mask,' sum = ',np.sum(sparse_forest_mask[1]),' len = ',len(sparse_forest_mask[1]))


            def area( a ):
                print(np.sum(a))
                xw=area_per_pixel*np.sum(a)
                print(xw)
                return xw/1000000
            d=['time','dfm','ofm','sfm','tfa']
            x=[]
            for i in range(len(dense_forest_mask)):
                w=[]
                w.append(pd.to_datetime(time_values[i]))
                w.append(area(dense_forest_mask[i]))
                w.append(area(open_forest_mask[i]))
                w.append(area(sparse_forest_mask[i]))
                w.append(area(dense_forest_mask[i])+area(open_forest_mask[i])+area(sparse_forest_mask[i]))
                x.append(w)
            # df['time'] = pd.to_datetime(time_values)
            df = pd.DataFrame(x, columns=d)
            df

            # Assuming your time column is named 'time' and the value column is named 'ndvi'
            # Convert the 'time' column to pandas timetime if it's not already in that format
            # Read the CSV file into a pandas DataFrame

            df['time'] = pd.to_datetime(df['time'])
            print(df.head())
            X_train=df['time']
            y_train=df['tfa']
            # Split the data into training and test sets
            # X_train, X_test, y_train, y_test = train_test_split(df['time'], df['dfm'],test_size=0.3,shuffle=False)

            # Extract the time components as features
            X_train_features = pd.DataFrame()
            X_train_features['year'] = X_train.dt.year
            X_train_features['month'] = X_train.dt.month
            X_train_features['day'] = X_train.dt.day
            # Add more features as per your requirements



        

            # Initialize and fit the Random Forest Regressor model
            model = RandomForestRegressor()
            model.fit(X_train_features, y_train)

            # Extract features from the test data
            X_test_features = pd.DataFrame()
            X_test_features['year'] = [2025]
            X_test_features['month'] =[ 5]
            X_test_features['day'] = [5]
            # Add more features as per your requirements
            print(X_train_features.head())
            print(X_test_features.head())
            # Predict the values
            predictions = model.predict(X_test_features)

            # Print the predictions
            print(predictions)


            prede=model.predict(X_train_features)
            print(prede)


            indices = np.arange(len(X_train))

            df['date'] = pd.to_datetime(df['time'])

            # Plot the actual values
            plt.plot(df['date'], df['tfa'], color='blue', label='Actual')

            # Plot the predicted values
            plt.plot(df['date'], prede, color='red', label='Predicted')

            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Forest Area(sq_km)')
            plt.title('Random Forest Predictions - Actual vs. Predicted'+' Selection-'+str(j+1)+' of color-'+c[j])
            j=j+1

            # Add legend
            plt.legend()

            # Display the graph
            # # plt.show()




            # Convert the plot to a PNG image in memory
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0) 

            plt.close()

            # Encode the PNG image as a base64 string
            i_base64 = base64.b64encode(buffer.read()).decode('utf-8')


            image_base64.append(i_base64)


            



            # import pandas as pd
            # import numpy as np
            # from sklearn.ensemble import RandomForestRegressor
            # import matplotlib.pyplot as plt



            # df['date'] = pd.to_datetime(df['time'])
            # X = df['date'].values.reshape(-1, 1)
            # y = df['tfa'].values



            # model = RandomForestRegressor()
            # model.fit(X, y)


            # present_dates = df['date']
            # future_dates = pd.date_range(start=present_dates.max(), periods=365)  # Change the number of periods as desired

            # print("--------------------------------------------------------------------------------------------------",future_dates)
            # present_predictions = model.predict(X)
            # future_predictions = model.predict(future_dates.values.reshape(-1, 1))


            # # plt.plot(present_dates, y, label='Actual')
            # plt.plot(present_dates, present_predictions, label='Present Predictions')
            # plt.plot(future_dates, future_predictions, label='Future Predictions')
            # plt.xlabel('Date')
            # plt.ylabel('Forest Area(sq_km)')
            # plt.legend()
            # plt.show()


            df['date'] = pd.to_datetime(df['time'])
            X = df['date'].values.reshape(-1, 1)
            y = df['tfa'].values

            model = RandomForestRegressor()
            model.fit(X, y)

            present_dates = df['date']
            num_months = 12  # Change the number of months as desired

            # Generate future dates
            start_date = present_dates.max().replace(day=1)
            future_dates = pd.date_range(start=start_date, periods=num_months, freq='MS')
            future_dates = future_dates.insert(0, present_dates.max())

            present_predictions = model.predict(X)
            future_predictions = model.predict(future_dates.values.reshape(-1, 1))

            # plt.plot(present_dates, y, label='Actual')
            plt.plot(present_dates, present_predictions, label='Present Predictions')
            plt.plot(future_dates, future_predictions, label='Future Predictions')
            plt.xlabel('Date')
            plt.ylabel('Forest Area(sq_km)')
            plt.title('Present Predictions with Future Predictions'+' Selection-'+str(k+1)+' of color-'+c[k])
            k=k+1
            plt.legend()

            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            # plt.show()  # Display the plot







            # Convert the plot to a PNG image in memory
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            plt.close()

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # Encode the PNG image as a base64 string
            h_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            hist_base64.append(h_base64)

           

        # if image_base64 is not None:
        #     return jsonify({'img_base64': image_base64})
        # else:
        #     return jsonify({'error': 'Invalid method selected.'})


    

        #return render_template('proj.html', img_base64=image_base64)

        # Return the base64-encoded image path as a JSON response
        # Append the plot_base64 to image_base64 array



    print(len(image_base64))
    print(len(hist_base64))

    if((len(lat_starts))==0):
        return jsonify({'l':"0"})
    else:
        if method == 'ndvi':
            return jsonify({'img_base64': image_base64,'his_base64':hist_base64})
        elif method == 'ndwi':
            return jsonify({'img_base64': image_base64,'his_base64':hist_base64})
        elif method == 'forest':
            return jsonify({'img_base64': image_base64})
        elif method == 'ml':
            return jsonify({'img_base64': image_base64,'his_base64':hist_base64})                                                    



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')



