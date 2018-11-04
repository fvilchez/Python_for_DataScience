########################## Caso de estudio ################################
import pandas as pd
from bokeh.io import output_file, show, curdoc
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, Slider, Select
import numpy as np
from bokeh.palettes import Spectral6
from bokeh.layouts import widgetbox, row

##### Cargamos los datos
data = pd.read_csv('gapminder_tidy.csv', index_col = 'Year')


###### Nos creamos nuestro objeto de tipo ColumnDataSource
##source = ColumnDataSource(data = {'x': data.loc[1970].fertility,
##                                  'y': data.loc[1970].life,
##                                  'country':data.loc[1970].Country})
##
##
###Nos creamos nuestra figura
##p = figure(title='1970', x_axis_label='Fertility (children per woman)',
##           y_axis_label='Life Expectancy (years)',plot_height=400,
##           plot_width=700,tools=[HoverTool(tooltips='@country')])
##
##
###Afregamos nuestro glyph
##p.circle(x = 'x', y = 'y', source = source)
##
##
###Nos creamos el fichero y vemos el resultado
##output_file('gapminder.html')
##show(p)

######################## Nos creamos un gráfico básico #########################

#Nos creamos nuestro elemento de tipo ColumnDataSource
source = ColumnDataSource(data = {
    'x': data.loc[1970, 'fertility'],
    'y': data.loc[1970, 'life'],
    'country': data.loc[1970, 'Country'],
    'pop': (data.loc[1970, 'population'] / 20000000) + 2,
    'region': data.loc[1970, 'region']})


#Calculamos el mínimo y el máximo de fertilidad y esperanza de vida
xmin, xmax = np.min(data.fertility), np.max(data.fertility)
ymin, ymax = np.min(data.life), np.max(data.life)

#Nos creamos nuestra figura
plot = figure(title = 'Gapminder Data for 1970', plot_height = 400, plot_width = 700,
              x_range = (xmin, xmax), y_range = (ymin, ymax))

##plot.circle(x = 'x', y = 'y', fill_alpha = 0.8, source = source)
##
##curdoc().add_root(plot)


######################## Hacemos uso de colormapper ############################

regions_list = data.region.unique().tolist()

color_mapper = CategoricalColorMapper(factors = regions_list,
                                      palette = Spectral6)

plot.circle(x = 'x', y = 'y', fill_alpha = 0.8, source = source,
            color = dict(field = 'region', transform = color_mapper),
            legend = 'region')



plot.legend.location = 'top_right'




########## Hacemos uso de un slidder para variar el año ########################
##def update_plot(attr, old, new):
##    yr = slider.value
##    new_data = {
##        'x': data.loc[yr, 'fertility'],
##        'y': data.loc[yr, 'life'],
##        'country': data.loc[yr, 'Country'],
##        'pop': (data.loc[yr, 'population'] / 20000000) + 2,
##        'region': data.loc[yr, 'region']}
##    source.data = new_data
##    #Hacemos el título de nuestro gráfico cambien
##    plot.title.text = 'Gapminder data for %d' % yr
##
##slider = Slider(start = 1970, end = 2010,  step = 1, value = 1970,
##                title = 'Year')
##
##hover = HoverTool(tooltips = [('Country', '@country')])
##
##plot.add_tools(hover)
##
##
##slider.on_change('value', update_plot)
##
##
##layout = row(widgetbox(slider), plot)
##curdoc().add_root(layout)
##curdoc().title = 'Gapminder'


######################## Agregamos más elementos ##############################

def update_plot(attr, old, new):
    #Leemos los valores de los elementos gráficos
    yr = slider.value
    x = x_select.value
    y = y_select.value
    #Fijamos los ejes
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    #Fijamos los nuevos datos
    new_data = {
        'x' : data.loc[yr, x],
        'y' : data.loc[yr, y],
        'country' : data.loc[yr, 'Country'],
        'pop' : (data.loc[yr, 'population'] / 20000000) + 2,
        'region' : data.loc[yr, 'region']}

    #Asignamos los nuevos datos
    source.data = new_data

    #Fijamos el rango de visualización de nuetros datos
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    #Creamos nuestro título del gráfico para que sea variale con el slider
    plot.title.text = 'Gapminder data for %d' % yr

#Nos creamos nuestro slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970,
                title = 'Year')
#Actualizamos nuestro slider
slider.on_change('value', update_plot)

#Nos creamos nuestro dropdown
x_select = Select(options = ['fertility', 'life', 'child_mortality', 'gdp'],
                  value = 'fertility',
                  title = 'x-axis-data')
#Actualizamos
x_select.on_change('value', update_plot)

#Nos creamos nuestro segundo dropdown
y_select = Select(options = ['fertility', 'life', 'child_mortality', 'gdp'],
                  value = 'life',
                  title = 'y-axis-data')
#Actualizamos
y_select.on_change('value', update_plot)

#Nos creamos nuestro layout
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)


    
    



