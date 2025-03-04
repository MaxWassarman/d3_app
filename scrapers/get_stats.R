install.packages(c("dplyr", "purrr", "cli", "remotes"), repos = "https://cloud.r-project.org")

remotes::install_url("https://github.com/BillPetti/baseballr/archive/refs/heads/master.tar.gz", 
                     dependencies = TRUE)
remotes::install_url("https://github.com/robert-frey/collegebaseball/archive/refs/heads/main.tar.gz", 
                     dependencies = TRUE)
# Load GitHub packages
library(dplyr)
library(purrr)
library(cli)
library(remotes)
library(baseballr)
library(collegebaseball)

data_dir <- "data"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

ncaa_stats_bulk <- function(year, 
                            type = 'batting', 
                            divisions = 3, 
                            situation = "all") {
  if (year < 2013) {
    stop('Year must be greater than or equal to 2013')
  }
  
  if (!type %in% c("batting", "pitching", "fielding")) {
    stop('Type must be "batting", "pitching", or "fielding"')
  }
  
  teams_lookup <- baseballr:::rds_from_url(
    "https://raw.githubusercontent.com/robert-frey/college-baseball/main/ncaa_team_lookup.rds"
  ) %>%
    dplyr::filter(year == !!year,
                  division %in% !!divisions) %>%
    distinct(team_id, .keep_all = TRUE)
  
  total_teams <- nrow(teams_lookup)
  cli::cli_alert_info(paste("Retrieving", type, "stats for", total_teams, "teams"))
  
  safe_ncaa_stats <- purrr::safely(ncaa_stats)
  
  results <- purrr::map(
    seq_len(nrow(teams_lookup)),
    function(i) {
      team <- teams_lookup[i,]
      
      if (i %% 10 == 0) {
        cli::cli_alert_info(paste("Processing team", i, "of", total_teams))
      }
      
      result <- safe_ncaa_stats(
        team_id = team$team_id,
        year = year,
        type = type,
        situation = situation
      )
      
      if (!is.null(result$error)) {
        cli::cli_alert_warning(paste("Error processing team_id:", team$team_id))
        return(NULL)
      }
      
      if (!is.null(result$result)) {
        result$result <- result$result %>%
          mutate(across(where(is.logical), as.character))
      }
      
      # Add a small delay to avoid overwhelming the NCAA server
      Sys.sleep(0.2)
      
      return(result$result)
    }
  )
  
  combined_stats <- results %>%
    purrr::compact() %>%
    dplyr::bind_rows()
  
  cli::cli_alert_success(paste("Retrieved stats for", 
                               nrow(combined_stats), 
                               "players across",
                               length(unique(combined_stats$team_id)),
                               "teams"))
  
  return(combined_stats)
}

# Get the current year
current_year <- as.integer(format(Sys.Date(), "%Y"))

# Run for current year
for (division in 3) {
  year <- current_year
  
  # Try to collect batting stats
  cli::cli_alert_info(paste("Collecting batting stats for D", division, year))
  tryCatch({
    batting <- ncaa_stats_bulk(year = year, type = "batting", divisions = division)
    write.csv(batting, file.path(data_dir, paste0("d", division, "_batting_", year, ".csv")), row.names = FALSE)
    cli::cli_alert_success(paste("Successfully saved batting stats for D", division, year))
  }, error = function(e) {
    cli::cli_alert_danger(paste("Failed to collect batting stats:", e$message))
  })
  
  # Try to collect pitching stats
  cli::cli_alert_info(paste("Collecting pitching stats for D", division, year))
  tryCatch({
    pitching <- ncaa_stats_bulk(year = year, type = "pitching", divisions = division)
    write.csv(pitching, file.path(data_dir, paste0("d", division, "_pitching_", year, ".csv")), row.names = FALSE)
    cli::cli_alert_success(paste("Successfully saved pitching stats for D", division, year))
  }, error = function(e) {
    cli::cli_alert_danger(paste("Failed to collect pitching stats:", e$message))
  })
}

# Output a completion message
cli::cli_alert_success(paste("Data collection completed at", Sys.time()))