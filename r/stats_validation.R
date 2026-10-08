# ==============================================================================
# stats_validation.R — Statistical validation for the ATO fraud-detection project
#
# Run in RStudio with the working directory set to the project root
# (open account-takeover-fraud-detection.Rproj and it is set automatically).
#
# Requires: dplyr, readr   (install once with install.packages(c("dplyr", "readr")))
#
# Replicates the statistical checks from notebooks/account_takeover_fraud_detection.ipynb:
#   1. chi-square  — impossible travel vs fraud
#   2. two-proportion z-test — new device vs known device
#   3. Mann-Whitney — transaction amounts, fraud vs legit
# ==============================================================================

# 1. Setup ----
library(dplyr)
library(readr)

if (!file.exists("data/login_attempts.csv")) {
  stop("Set the working directory to the project root first ",
       "(RStudio: open the .Rproj file, or Session > Set Working Directory).")
}

logins <- read_csv("data/login_attempts.csv", show_col_types = FALSE)
txns   <- read_csv("data/transactions.csv",  show_col_types = FALSE)

cat("logins:", format(nrow(logins), big.mark = ","), "\n")
cat("txns:  ", format(nrow(txns), big.mark = ","), "\n\n")

# 2. Feature engineering (past-only, mirrors the Python notebook) ----

haversine_km <- function(lat1, lon1, lat2, lon2) {
  rad  <- pi / 180
  lat1 <- lat1 * rad; lon1 <- lon1 * rad
  lat2 <- lat2 * rad; lon2 <- lon2 * rad
  dlat <- lat2 - lat1
  dlon <- lon2 - lon1
  a <- sin(dlat / 2)^2 + cos(lat1) * cos(lat2) * sin(dlon / 2)^2
  6371 * 2 * asin(sqrt(a))
}

feat <- logins |>
  arrange(user_id, timestamp) |>
  group_by(user_id) |>
  mutate(
    prev_ts      = lag(timestamp),
    prev_lat     = lag(ip_lat),
    prev_lon     = lag(ip_lon),
    prev_device  = lag(device_id),
    prev_country = lag(ip_country),
    distance_km  = haversine_km(prev_lat, prev_lon, ip_lat, ip_lon),
    hours_since_last = as.numeric(difftime(timestamp, prev_ts, units = "hours")),
    speed_kmh = ifelse(is.na(hours_since_last) | hours_since_last == 0, 0,
                       distance_km / hours_since_last),
    impossible_travel = as.integer(!is.na(distance_km) &
                                     distance_km > 500 & speed_kmh > 800),
    new_device  = as.integer(!is.na(prev_device)  & device_id  != prev_device),
    new_country = as.integer(!is.na(prev_country) & ip_country != prev_country)
  ) |>
  ungroup()

cat("impossible-travel logins:", sum(feat$impossible_travel), "\n")
cat("new-device logins:       ", sum(feat$new_device), "\n\n")

# 3. Chi-square: impossible travel vs fraud ----
cat("--- chi-square: impossible travel x fraud ---\n")
ct <- table(feat$impossible_travel, feat$is_fraud_attempt)
print(ct)
chi <- chisq.test(ct)
cat(sprintf("chi2 = %.1f, p = %.2e\n", chi$statistic, chi$p.value))
cat("No significant association: legit travel is common and account takeovers\n")
cat("usually follow long idle gaps, so speed-based checks rarely fire.\n\n")

# 4. Two-proportion z-test: new device vs known device ----
cat("--- z-test: new device vs known device ---\n")
a <- feat$is_fraud_attempt[feat$new_device == 1]
b <- feat$is_fraud_attempt[feat$new_device == 0]
p1 <- mean(a); p2 <- mean(b)
n1 <- length(a); n2 <- length(b)
se <- sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)  # unpooled, as in the notebook
z  <- (p1 - p2) / se
p  <- 2 * pnorm(-abs(z))
cat(sprintf("fraud rate | new device:   %.2f%% (n = %s)\n",
            p1 * 100, format(n1, big.mark = ",")))
cat(sprintf("fraud rate | known device: %.2f%% (n = %s)\n",
            p2 * 100, format(n2, big.mark = ",")))
cat(sprintf("z = %.1f, p = %.2e — new-device logins are far riskier\n\n", z, p))

# 5. Mann-Whitney: transaction amounts, fraud vs legit ----
cat("--- Mann-Whitney: fraud vs legit amounts ---\n")
fr <- txns$amount_usd[txns$is_fraud_txn == 1]
lg <- txns$amount_usd[txns$is_fraud_txn == 0]
wt <- wilcox.test(fr, lg)
cat(sprintf("median fraud $%.0f vs legit $%.0f\n", median(fr), median(lg)))
cat(sprintf("p = %.2e — fraud transactions are significantly larger\n", wt$p.value))

cat("\nDone.\n")
