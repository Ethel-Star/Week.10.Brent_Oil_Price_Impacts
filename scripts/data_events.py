import pandas as pd

import os
import matplotlib.pyplot as plt

def create_events_data(output_dir):
    """
    Create a DataFrame of key events affecting Brent oil prices from 1987–2022 and save it to CSV.
    
    :param output_dir: Directory to save the events CSV file.
    :return: DataFrame with events data.
    """
    events_data = {
        'Date': [
            # 1987–1996 (20 events)
            '1987-07-01', '1988-07-18', '1989-02-15', '1989-12-20', '1990-08-02',
            '1990-10-01', '1991-01-17', '1991-03-01', '1991-06-01', '1992-06-01',
            '1993-03-01', '1994-03-01', '1994-11-01', '1995-03-01', '1995-06-01',
            '1996-03-01', '1996-06-01', '1996-09-01', '1996-11-01', '1996-12-01',
            # 1997–2006 (30 events)
            '1997-03-01', '1997-06-25', '1997-11-01', '1998-03-25', '1998-06-24',
            '1998-12-01', '1999-03-23', '1999-06-01', '2000-03-28', '2000-06-21',
            '2000-09-01', '2001-01-17', '2001-09-11', '2002-01-01', '2002-06-26',
            '2003-01-12', '2003-03-20', '2003-06-01', '2004-05-01', '2004-08-01',
            '2005-01-30', '2005-06-15', '2005-08-29', '2005-12-12', '2006-03-08',
            '2006-07-12', '2006-10-19', '2006-12-14', '2007-03-01', '2007-06-01',
            # 2007–2016 (30 events)
            '2007-09-11', '2007-12-05', '2008-03-05', '2008-07-11', '2008-09-15',
            '2008-10-24', '2008-12-17', '2009-03-01', '2009-05-28', '2010-04-20',
            '2010-10-14', '2011-02-15', '2011-06-08', '2011-12-14', '2012-03-01',
            '2012-07-01', '2013-03-01', '2013-07-01', '2014-03-01', '2014-06-01',
            '2014-11-27', '2015-06-05', '2015-12-04', '2015-12-18', '2016-04-17',
            '2016-09-28', '2016-11-30', '2017-03-01', '2017-05-25', '2017-06-01',
            # 2017–2022 (20 events)
            '2017-11-30', '2018-05-08', '2018-06-22', '2018-11-05', '2018-12-07',
            '2019-03-01', '2019-07-02', '2019-12-05', '2020-01-03', '2020-03-06',
            '2020-03-11', '2020-04-12', '2020-06-06', '2020-12-03', '2021-03-04',
            '2021-07-18', '2021-10-04', '2022-02-24', '2022-06-02', '2022-09-05'
        ],
        'Event_Type': [
            # 1987–1996
            'OPEC Policy', 'Conflict Resolution', 'Sanctions', 'Political Decision', 'Conflict',
            'Sanctions', 'Conflict', 'Conflict Resolution', 'OPEC Policy', 'OPEC Policy',
            'Sanctions', 'Sanctions', 'OPEC Policy', 'Political Decision', 'OPEC Policy',
            'Sanctions', 'OPEC Policy', 'Political Decision', 'OPEC Policy', 'Sanctions',
            # 1997–2006
            'OPEC Policy', 'OPEC Policy', 'Sanctions', 'OPEC Policy', 'OPEC Policy',
            'OPEC Policy', 'OPEC Policy', 'Political Decision', 'OPEC Policy', 'OPEC Policy',
            'Sanctions', 'OPEC Policy', 'Conflict', 'OPEC Policy', 'OPEC Policy',
            'OPEC Policy', 'Conflict', 'Political Decision', 'Sanctions', 'Political Decision',
            'OPEC Policy', 'OPEC Policy', 'Political Decision', 'OPEC Policy', 'Sanctions',
            'Conflict', 'OPEC Policy', 'OPEC Policy', 'Sanctions', 'Political Decision',
            # 2007–2016
            'OPEC Policy', 'OPEC Policy', 'Sanctions', 'Economic', 'Economic',
            'OPEC Policy', 'OPEC Policy', 'OPEC Policy', 'Political Decision', 'Political Decision',
            'OPEC Policy', 'Conflict', 'OPEC Policy', 'OPEC Policy', 'Sanctions',
            'Sanctions', 'Political Decision', 'Conflict', 'Sanctions', 'Sanctions',
            'OPEC Policy', 'OPEC Policy', 'OPEC Policy', 'Political Decision', 'OPEC Policy',
            'OPEC Policy', 'OPEC Policy', 'Sanctions', 'OPEC Policy', 'Political Decision',
            # 2017–2022
            'OPEC Policy', 'Sanctions', 'OPEC Policy', 'Sanctions', 'OPEC Policy',
            'Sanctions', 'OPEC Policy', 'OPEC Policy', 'Conflict', 'OPEC Policy',
            'Economic', 'OPEC Policy', 'OPEC Policy', 'OPEC Policy', 'OPEC Policy',
            'OPEC Policy', 'OPEC Policy', 'Conflict', 'OPEC Policy', 'OPEC Policy'
        ],
        'Event_Description': [
            # 1987–1996
            'OPEC stabilizes prices post-1986 glut with quota enforcement',
            'Iran-Iraq War ends; oil supply stabilizes',
            'U.S. sanctions on Libya tightened',
            'U.S. Strategic Petroleum Reserve release announced',
            'Iraq invades Kuwait; Gulf War begins, oil supply fears spike',
            'UN sanctions imposed on Iraq post-invasion',
            'Gulf War air campaign begins; oil prices surge',
            'Gulf War ends; Iraqi oil supply resumes gradually',
            'OPEC increases production quotas post-Gulf War',
            'OPEC adjusts quotas to stabilize market',
            'UN extends Iraq sanctions, oil exports limited',
            'UN tightens sanctions on Iraq, limiting oil exports',
            'OPEC cuts production to support prices',
            'U.S. energy policy shifts favor domestic production',
            'OPEC increases output to meet demand',
            'UN imposes sanctions on Libya over Lockerbie',
            'OPEC adjusts quotas amid stable prices',
            'U.S. relaxes some oil import policies',
            'OPEC maintains production levels',
            'UN sanctions on Iraq persist, oil supply constrained',
            # 1997–2006
            'OPEC increases production quotas',
            'OPEC cuts production to counter low prices',
            'UN sanctions on Iraq adjusted under Oil-for-Food',
            'OPEC cuts production amid Asian financial crisis',
            'OPEC agrees to further cuts amid low prices',
            'OPEC boosts production to stabilize prices',
            'OPEC raises output to meet demand',
            'U.S. energy policy encourages shale development',
            'OPEC increases production quotas',
            'OPEC boosts output amid high prices',
            'U.S. sanctions on Iran tightened',
            'OPEC cuts production to stabilize prices',
            '9/11 attacks; oil prices spike due to uncertainty',
            'OPEC increases output to calm markets',
            'OPEC adjusts quotas to balance supply',
            'OPEC cuts production amid Iraq tensions',
            'U.S. invades Iraq; fears of oil supply disruption',
            'Iraq oil production resumes post-war',
            'UN lifts Iraq sanctions, oil exports rise',
            'U.S. energy policy shifts post-Iraq',
            'OPEC increases quotas amid high demand',
            'OPEC cuts production to support prices',
            'Hurricane Katrina disrupts U.S. Gulf oil production',
            'OPEC maintains output despite high prices',
            'U.S. imposes sanctions on Iran nuclear program',
            'Israel-Hezbollah War; Middle East tensions rise',
            'OPEC cuts production by 1 million bpd',
            'OPEC reduces output to counter oversupply',
            'UN sanctions on Iran escalate',
            'U.S. energy policy boosts biofuels',
            # 2007–2016
            'OPEC increases production quotas',
            'OPEC maintains output amid high prices',
            'U.S. sanctions on Iran tightened further',
            'Global financial crisis peaks; oil hits $147 then crashes',
            'Lehman Brothers collapse; oil demand plummets',
            'OPEC cuts production by 2.2 million bpd',
            'OPEC cuts production by 4.2 million bpd to counter crisis',
            'OPEC maintains output as prices recover',
            'U.S. energy policy shifts toward renewables',
            'Deepwater Horizon spill; U.S. Gulf production halted',
            'OPEC adjusts quotas amid recovery',
            'Arab Spring begins; Libyan oil production halted',
            'OPEC fails to reach production agreement',
            'OPEC maintains output amid Libyan recovery',
            'U.S. sanctions on Iran escalate',
            'EU sanctions on Iran over nuclear program begin',
            'U.S. energy policy boosts shale production',
            'Egypt unrest escalates; oil transit fears',
            'U.S. imposes new sanctions on Russia',
            'Russia annexes Crimea; Western sanctions imposed',
            'OPEC maintains output despite oversupply; prices crash',
            'OPEC maintains high production levels',
            'OPEC fails to cut production; prices drop',
            'U.S. lifts crude oil export ban, boosting supply',
            'OPEC+ Doha talks fail; no production freeze',
            'OPEC agrees to modest production cut in Algiers',
            'OPEC agrees to cut 1.2 million bpd; prices rise',
            'Iran sanctions eased under nuclear deal',
            'OPEC extends production cuts',
            'U.S. energy policy shifts under Trump',
            # 2017–2022
            'OPEC extends production cuts again',
            'U.S. withdraws from Iran nuclear deal; sanctions loom',
            'OPEC increases production to offset Iran losses',
            'U.S. reimposes sanctions on Iran; oil supply tightens',
            'OPEC+ agrees to cut production by 1.2 million bpd',
            'U.S. sanctions on Venezuela escalate',
            'OPEC+ extends production cuts',
            'OPEC+ agrees to further cuts amid demand fears',
            'U.S. drone strike kills Iran’s Soleimani; oil spikes',
            'OPEC+ talks fail; Saudi-Russia price war begins',
            'COVID-19 declared pandemic; oil demand collapses',
            'OPEC+ agrees to historic 9.7 million bpd cut',
            'OPEC+ extends deep cuts to July',
            'OPEC+ begins phasing out cuts',
            'OPEC+ increases output amid recovery',
            'OPEC+ agrees to gradual production increase',
            'OPEC+ boosts output as prices rise',
            'Russia invades Ukraine; oil prices surge due to supply fears',
            'OPEC+ increases production amid high prices',
            'OPEC+ cuts production by 2 million bpd'
        ]
    }

    # Create DataFrame
    events = pd.DataFrame(events_data)
    events['Date'] = pd.to_datetime(events['Date'])

    # Save to CSV
    output_path = os.path.join(output_dir, 'key_events_1987_2022.csv')
    events.to_csv(output_path, index=False)
    print(f"Events saved to '{output_path}'")

    return events
def major_events():
   major_event_dates = [
    '1987-07-01',  # OPEC stabilizes post-glut
    '1990-08-02',  # Gulf War begins
    '1991-03-01',  # Gulf War ends
    '1998-12-01',  # OPEC cuts amid Asian crisis
    '2001-09-11',  # 9/11 attacks
    '2003-03-20',  # U.S. invades Iraq
    '2008-07-11',  # Financial crisis peak
    '2008-12-17',  # OPEC cuts 4.2M bpd
    '2011-02-15',  # Arab Spring begins
    '2014-11-27',  # OPEC maintains output; prices crash
    '2016-11-30',  # OPEC cuts 1.2M bpd
    '2018-11-05',  # U.S. reimposes Iran sanctions
    '2020-03-11',  # COVID-19 pandemic declared
    '2020-04-12',  # OPEC+ historic 9.7M bpd cut
    '2022-02-24'   # Russia invades Ukraine
  ]
   return major_event_dates
# Filter major events
def plot_price_events(merged_data, figsize=(14, 7), price_color='blue', event_color='red',
                     annot_size=8, annot_length=20, grid_style=True):
    """
    Visualize Brent oil prices with event annotations
    
    Parameters:
    merged_data (pd.DataFrame): DataFrame containing 'Date', 'Price', and event columns
    figsize (tuple): Figure dimensions (width, height) in inches
    price_color (str): Color for price line
    event_color (str): Color for event markers
    annot_size (int): Font size for annotations
    annot_length (int): Max characters for annotation text
    grid_style (bool): Whether to show grid
    
    Returns:
    matplotlib.figure.Figure: The generated figure object
    """
    # Validate input columns
    required_cols = ['Date', 'Price', 'Event_Type', 'Event_Description']
    if not all(col in merged_data.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Needed: {required_cols}")

    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price history
    ax.plot(merged_data['Date'], merged_data['Price'], 
            label='Brent Price', color=price_color)
    
    # Plot events
    event_days = merged_data[merged_data['Event_Type'] != 'None']
    ax.scatter(event_days['Date'], event_days['Price'], 
               color=event_color, label='Events', zorder=5)
    
    # Add annotations
    for _, row in event_days.iterrows():
        ax.annotate(
            text=row['Event_Description'][:annot_length] + "...",
            xy=(row['Date'], row['Price']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=annot_size,
            alpha=0.8
        )
    
    # Formatting
    ax.set_title('Brent Oil Prices with Major Events (1987–2022)', pad=20)
    ax.set_xlabel('Date', labelpad=10)
    ax.set_ylabel('Price (USD/barrel)', labelpad=10)
    ax.legend(loc='upper left')
    
    if grid_style:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig